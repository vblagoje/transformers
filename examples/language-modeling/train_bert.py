# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import logging
import os
import random
from copy import copy
from typing import Dict, List, Optional, Any

import datasets
import nltk
from dataclasses import dataclass, field
from datasets import load_dataset, load_from_disk
from pytorch_block_sparse import BlockSparseModelPatcher
from torch.utils.data.dataset import Dataset

from transformers import (
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed, BertConfig, BertTokenizerFast, BertForPreTraining, DataCollatorForNextSentencePrediction, TrainerState,
)

nltk.download('punkt')

logger = logging.getLogger(__name__)


class DatasetAdapter(Dataset):

    def __init__(self, dataset: datasets.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]


class AndBNextSentencePredictionProcessor(object):

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            block_size: int = 128,
            short_seq_probability=0.1,
            nsp_probability=0.5,
            text_column: str = "text"
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability
        self.text_column: str = text_column

    def __call__(self, documents: List[str]) -> Dict[str, Any]:
        examples_a_all, examples_b_all, examples_is_random_next_all = [], [], []
        encoded_docs = []
        for document in documents[self.text_column]:
            encoded_doc = self.encode_document(document)
            encoded_docs.append(encoded_doc)

        for idx, doc in enumerate(encoded_docs):
            examples_a, examples_b, examples_is_random_next = self.create_examples_from_document(encoded_docs, doc, idx)
            examples_a_all.extend(examples_a)
            examples_b_all.extend(examples_b)
            examples_is_random_next_all.extend(examples_is_random_next)

        return {"is_random_next": examples_is_random_next_all,
                "tokens_a": examples_a_all,
                "tokens_b": examples_b_all}

    def encode_document(self, text: str) -> List[List[int]]:
        sentences = nltk.sent_tokenize(text)
        batch_encoding = self.tokenizer.batch_encode_plus(sentences, add_special_tokens=False,
                                                          truncation=True, max_length=self.block_size)
        return batch_encoding["input_ids"]

    def create_examples_from_document(self, documents: List[List[List[int]]],
                                      document: List[List[int]], doc_idx: int):

        """Creates examples for a single document."""
        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=True)

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pre-training and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)

        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0

        examples_a, examples_b, examples_is_random_next = [], [], []
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    if len(current_chunk) == 1 or random.random() < self.nsp_probability:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index = random.randint(0, len(documents) - 1)
                            if random_document_index != doc_idx:
                                break

                        random_document = documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # Actual next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    examples_a.append(tokens_a)
                    examples_b.append(tokens_b)
                    examples_is_random_next.append(is_random_next)

                current_chunk = []
                current_length = 0

            i += 1
        return examples_a, examples_b, examples_is_random_next


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name: Optional[str] = field(
        default="bert-tiny",
        metadata={"help": "Model name [bert-tiny, bert-mini, bert-small, bert=medium, bert-base]"},
    )
    tokenizer_name: Optional[str] = field(
        default='bert-base-uncased',
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    sparse_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Turn on/off model sparsification"}
    )

    sparse_density: Optional[float] = field(
        default=0.25,
        metadata={"help": "Sparsification density"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training
    """
    encoded_bert_dataset_path: Optional[str] = field(
        default="encoded_bert_dataset",
        metadata={"help": "An optional parameter specifying path to encoded bert dataset to use in training."},
    )

    block_size: int = field(
        default=512,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into "
                    "account special tokens)."
        },
    )


@dataclass
class BertTrainingArguments(TrainingArguments):
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using :class:`~transformers.HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on the command line.
    """

    warmup_proportion: Optional[float] = field(
        default=0.1,
        metadata={"help": "Proportion of training steps for warmup"}
    )

    warmup_proportion_phase2: Optional[float] = field(
        default=0.1,
        metadata={"help": "Proportion of training steps for warmup"}
    )

    learning_rate_phase2: Optional[float] = field(
        default=4e-3,
        metadata={"help": "The initial learning rate for phase two"}
    )

    gradient_accumulation_steps_phase2: Optional[int] = field(
        default=32,
        metadata={"help": "Number of updates steps to accumulate gradients before performing a backward/update pass."},
    )

    max_steps_phase2: Optional[int] = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )

    per_device_train_batch_size_phase2: Optional[int] = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

    do_phase1_training: Optional[bool] = field(
        default=False,
        metadata={"help": "If true phase 1 training will be executed"}
    )

    do_phase2_training: Optional[bool] = field(
        default=False,
        metadata={"help": "If true phase 2 training will be executed"}
    )


def find_checkpoint(args: TrainingArguments):
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_dir)
    if checkpoints:
        checkpoint_dir = checkpoints[0]
    else:
        checkpoint_dir = None
    return checkpoint_dir


def prepare_bert_dataset(tokenizer: PreTrainedTokenizer, args: DataTrainingArguments):
    try:
        encoded_bert_dataset = load_from_disk(args.encoded_bert_dataset_path)
        logger.info(f"Using dataset of {len(encoded_bert_dataset)} samples")
    except FileNotFoundError:
        logger.info(f"Encoded bert dataset not found on path {args.encoded_bert_dataset_path}")
        bert_dataset = load_dataset('wikipedia', "20200501.en", split='train')
        nsp = AndBNextSentencePredictionProcessor(tokenizer=tokenizer)
        logger.info(f"Encoding bert dataset containing {len(bert_dataset)} articles")
        encoded_bert_dataset = bert_dataset.map(nsp, batched=True, remove_columns=bert_dataset.column_names)

    return encoded_bert_dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, BertTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Let's define only the main models using L/H tuples
    # <p>BERT Miniatures is the set of 24 BERT models referenced in
    # <a href="https://arxiv.org/abs/1908.08962">Well-Read Students Learn Better: On the Importance of
    # Pre-training Compact Models</a> (English only, uncased, trained with WordPiece masking).</p>
    #
    # The L and H in the table below stand for the numbers of transformer layers (L) and hidden
    # embedding sizes (H). The numberof self-attention heads is set to H/64 and the feed-forward/filter
    # size to 4*H.
    bert_configs = {"bert-tiny": (2, 128),
                    "bert-mini": (4, 256),
                    "bert-small": (4, 512),
                    "bert-medium": (8, 512),
                    "bert-base": (12, 768)}

    # and select the model to train
    l = bert_configs[model_args.model_name][0]
    h = bert_configs[model_args.model_name][1]
    config = BertConfig(hidden_size=h, num_attention_heads=int(h / 64), num_hidden_layers=l,
                        intermediate_size=4 * h)

    logger.info(f"Instantiating a new config instance {config}")

    # Fast tokenizer helps a lot with the speed of encoding
    tokenizer = BertTokenizerFast.from_pretrained(model_args.tokenizer_name)

    # fresh MLM/NSP model
    model = BertForPreTraining(config)

    if model_args.sparse_model:
        model = model.cuda()  # for some reason we have to do sparsification on cuda
        mp = BlockSparseModelPatcher()
        mp.add_pattern("bert\.encoder\.layer\.[0-9]+\.intermediate\.dense", {"density": model_args.sparse_density})
        mp.add_pattern("bert\.encoder\.layer\.[0-9]+\.output\.dense", {"density": model_args.sparse_density})
        mp.patch_model(model)
        logger.info(f"Model sparsified and now has {model.num_parameters()} parameters")

    logger.info("Preparing bert training dataset...")
    dataset = prepare_bert_dataset(tokenizer, data_args)

    # never mind the name, we are just splitting bert dataset into two parts
    bert_training_dataset = dataset.train_test_split(train_size=0.9, test_size=0.1)
    first_bert_training_dataset = bert_training_dataset["train"]
    second_bert_training_dataset = bert_training_dataset["test"]

    logger.info(f"First phase training dataset has {len(first_bert_training_dataset)} sequence pairs.")
    logger.info(f"Second phase training dataset has {len(second_bert_training_dataset)} sequence pairs.")

    first_phase_training_args = copy(training_args)
    second_phase_training_args = copy(training_args)

    first_phase_training_args.warmup_steps = int(first_phase_training_args.max_steps *
                                                 first_phase_training_args.warmup_proportion)

    # Training
    if training_args.do_train:

        if training_args.do_phase1_training:
            checkpoint_dir = find_checkpoint(training_args)
            if checkpoint_dir:
                logger.info(f"Training continues from checkpoint {checkpoint_dir}")
                model = model.from_pretrained(checkpoint_dir)
                model.train()
                logger.info(f"Loaded model with {model.num_parameters()} parameters from checkpoint")
            else:
                logger.info(f"Training starts from scratch")

            trainer = Trainer(model=model, args=first_phase_training_args,
                              data_collator=DataCollatorForNextSentencePrediction(tokenizer=tokenizer, block_size=128),
                              train_dataset=DatasetAdapter(first_bert_training_dataset))
            # checkpoint_dir is ignored if None
            trainer.train(model_path=checkpoint_dir)
            trainer.state.save_to_json(json_path=os.path.join(training_args.output_dir, "trainer_state.json"))

        if training_args.do_phase2_training:
            checkpoint_dir = find_checkpoint(training_args)
            if checkpoint_dir and not training_args.do_phase1_training:
                logger.info(f"Training phase 2 continues from checkpoint {checkpoint_dir}, please wait...")
                model = model.from_pretrained(checkpoint_dir)
                model.train()
                logger.info(f"Loaded model with {model.num_parameters()} parameters from checkpoint")

            trainer = Trainer(model=model,
                              args=prepare_training_args(copy(training_args), second_phase_training_args),
                              data_collator=DataCollatorForNextSentencePrediction(tokenizer=tokenizer,
                                                                                  block_size=512),
                              train_dataset=DatasetAdapter(second_bert_training_dataset))
            # adjust optimizer/scheduler for second phase
            trainer.create_optimizer_and_scheduler(num_training_steps=second_phase_training_args.max_steps_phase2)

            trainer_state = TrainerState.load_from_json(json_path=os.path.join(training_args.output_dir,
                                                                               "trainer_state.json"))
            logger.info(f"Starting phase 2, parameters {trainer.args}, global step is {trainer_state.global_step}")
            trainer.train(model_path=training_args.output_dir if trainer_state else None)

        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)


def prepare_training_args(training_args: TrainingArguments, second_phase_training_args: TrainingArguments):
    training_args.warmup_steps = int(second_phase_training_args.max_steps_phase2 *
                                     second_phase_training_args.warmup_proportion_phase2)
    training_args.learning_rate = second_phase_training_args.learning_rate_phase2
    training_args.max_steps += second_phase_training_args.max_steps_phase2
    training_args.gradient_accumulation_steps = second_phase_training_args.gradient_accumulation_steps_phase2
    training_args.per_device_train_batch_size = second_phase_training_args.per_device_train_batch_size_phase2
    training_args.run_name = training_args.run_name + "_2_phase"
    return training_args


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
