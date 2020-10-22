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
from copy import copy
from typing import Dict, List, Optional, Callable, Tuple, Union

import datasets
import nltk
import torch
from apex.optimizers import FusedLAMB
from dataclasses import dataclass, field
from datasets import load_from_disk
from pytorch_block_sparse import BlockSparseModelPatcher
from torch.utils.data.dataset import Dataset

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed, BertConfig, BertTokenizerFast, BertForPreTraining, DataCollatorForNextSentencePrediction, TrainerState,
    TrainerCallback, EvalPrediction, PreTrainedModel, DataCollator, get_polynomial_decay_schedule_with_warmup, )

nltk.download('punkt')

logger = logging.getLogger(__name__)


class DatasetAdapter(Dataset):

    def __init__(self, dataset: datasets.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]


class BertTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            **kwargs,
    ):
        super(BertTrainer, self).__init__(model, args, data_collator, train_dataset,
                                          eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers,
                                          **kwargs)


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


def prepare_training_args(training_args: TrainingArguments, second_phase_training_args: BertTrainingArguments):
    training_args.warmup_steps = int(second_phase_training_args.max_steps_phase2 *
                                     second_phase_training_args.warmup_proportion_phase2)
    training_args.learning_rate = second_phase_training_args.learning_rate_phase2
    training_args.max_steps += second_phase_training_args.max_steps_phase2
    training_args.gradient_accumulation_steps = second_phase_training_args.gradient_accumulation_steps_phase2
    training_args.per_device_train_batch_size = second_phase_training_args.per_device_train_batch_size_phase2
    training_args.run_name = training_args.run_name + "_2_phase"
    return training_args


def prepare_optimizer_and_scheduler(model, args) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMB(optimizer_grouped_parameters,
                          lr=args.learning_rate)
    # lr_scheduler = PolyWarmUpScheduler(optimizer,
    #                                    warmup=args.warmup_proportion,
    #                                    total_steps=args.max_steps)
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
                                                             num_warmup_steps=int(
                                                                 args.warmup_proportion * args.max_steps),
                                                             num_training_steps=args.max_steps)
    return optimizer, lr_scheduler


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
        num_parameters = model.num_parameters()
        model = model.cuda()  # for some reason we have to do sparsification on cuda
        mp = BlockSparseModelPatcher()
        mp.add_pattern("bert\.encoder\.layer\.[0-9]+\.intermediate\.dense", {"density": model_args.sparse_density})
        mp.add_pattern("bert\.encoder\.layer\.[0-9]+\.output\.dense", {"density": model_args.sparse_density})
        mp.patch_model(model)
        logger.info(f"Model sparsified from {num_parameters} parameters, now has {model.num_parameters()} parameters")

    logger.info("Preparing bert training dataset...")
    dataset = load_from_disk(data_args.encoded_bert_dataset_path)
    logger.info(f"Using dataset of {len(dataset)} samples")

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
                              data_collator=DataCollatorForNextSentencePrediction(tokenizer=tokenizer,
                                                                                  block_size=128),
                              train_dataset=DatasetAdapter(first_bert_training_dataset),
                              optimizers=prepare_optimizer_and_scheduler(model, first_phase_training_args))
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
                              train_dataset=DatasetAdapter(second_bert_training_dataset),
                              optimizers=prepare_optimizer_and_scheduler(model, second_phase_training_args))

            # adjust optimizer/scheduler for second phase
            trainer.create_optimizer_and_scheduler(num_training_steps=second_phase_training_args.max_steps_phase2)

            trainer_state = TrainerState.load_from_json(json_path=os.path.join(training_args.output_dir,
                                                                               "trainer_state.json"))
            logger.info(f"Starting phase 2, parameters {trainer.args}, global step is {trainer_state.global_step}")
            trainer.train(model_path=training_args.output_dir if trainer_state else None)

        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
