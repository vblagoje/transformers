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
from typing import Optional, Tuple, List, Dict, Union, Callable

import datasets
import torch
from apex.optimizers import FusedLAMB
from dataclasses import dataclass, field
from datasets import load_from_disk, concatenate_datasets
from pytorch_block_sparse import BlockSparseModelPatcher
from torch.utils.data import RandomSampler, Dataset, DataLoader, BatchSampler, SequentialSampler

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments, set_seed, BertConfig, BertForPreTraining,
    get_polynomial_decay_schedule_with_warmup, PreTrainedModel, DataCollator, EvalPrediction, TrainerCallback, )

logger = logging.getLogger(__name__)


class DatasetAdapter(torch.utils.data.Dataset):

    def __init__(self, dataset: datasets.Dataset):
        self.dataset = dataset
        self.dataset.set_format(**{'type': 'torch', 'format_kwargs': {'dtype': torch.long}})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if isinstance(index, list):
            first_index = index[0]
            last_index = index[-1]
            return self.dataset[first_index:last_index + 1]
        else:
            return self.dataset[index]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name: Optional[str] = field(
        default="bert-tiny",
        metadata={"help": "Model name [bert-tiny, bert-mini, bert-small, bert=medium, bert-base]"},
    )

    config_file: Optional[str] = field(
        default=None,
        metadata={"help": "Model config file"},
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
    input_dir: Optional[str] = field(
        default="phase1",
        metadata={"help": "Path to input data directory"},
    )


@dataclass
class BertTrainingArguments(TrainingArguments):
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using :class:`~transformers.HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on the command line.
    """

    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to load the checkpoint and resume training"}
    )

    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to resume training from checkpoint"}
    )

    phase1_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for phase1, used to load model for phase2"}
    )

    warmup_proportion: Optional[float] = field(
        default=0.1,
        metadata={"help": "Proportion of training steps for warmup"}
    )

    phase1: Optional[bool] = field(
        default=False,
        metadata={"help": "If true phase 1 training will be executed"}
    )

    phase2: Optional[bool] = field(
        default=False,
        metadata={"help": "If true phase 2 training will be executed"}
    )

    num_workers: Optional[int] = field(
        default=0,
        metadata={"help": "Dataloader's num_workers attribute"}
    )

    pin_memory: Optional[bool] = field(
        default=False,
        metadata={"help": "Dataloader's pin_memory attribute"}
    )


class BertTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module] = None,
            args: BertTrainingArguments = None,
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

    def get_train_dataloader(self) -> DataLoader:
        t_dataset = DatasetAdapter(self.train_dataset)
        return DataLoader(dataset=t_dataset,
                          sampler=BatchSampler(
                              SequentialSampler(t_dataset), batch_size=self.args.train_batch_size, drop_last=False
                          ),
                          collate_fn=lambda x: x[0],
                          pin_memory=self.args.pin_memory,
                          num_workers=self.args.num_workers)


def sparsify_model(model: PreTrainedModel, args: ModelArguments) -> PreTrainedModel:
    model = model.cuda()  # for some reason we have to do sparsification on cuda
    mp = BlockSparseModelPatcher()
    mp.add_pattern("bert\.encoder\.layer\.[0-9]+\.intermediate\.dense", {"density": args.sparse_density})
    mp.add_pattern("bert\.encoder\.layer\.[0-9]+\.output\.dense", {"density": args.sparse_density})
    mp.patch_model(model)
    return model


def get_bert_model_config(config_name: str) -> BertConfig:
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
    l = bert_configs[config_name][0]
    h = bert_configs[config_name][1]
    return BertConfig(hidden_size=h, num_attention_heads=int(h / 64), num_hidden_layers=l,
                      intermediate_size=4 * h)


def get_world_size(args):
    world_size = 1
    if args.local_rank != -1 and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    return max(1, world_size)


def prepare_optimizer_and_scheduler(model, args) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    m_params = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in m_params if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in m_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMB(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
                                                             num_warmup_steps=args.warmup_steps,
                                                             num_training_steps=args.max_steps,
                                                             power=0.5)
    return optimizer, lr_scheduler


def find_datasets_for_shard(input_dir, rank):
    shard_dataset_path = "_".join(["*", "shard", str(rank)])
    dir_path = os.path.join(input_dir, shard_dataset_path)
    return glob.glob(dir_path)


def load_dataset(data_args, rank):
    dataset_dirs = find_datasets_for_shard(data_args.input_dir, rank)
    if len(dataset_dirs) == 0:
        raise RuntimeError(f"No datasets found for process with {rank} in {data_args.input_dir}")
    elif len(dataset_dirs) >= 1:
        dataset = load_from_disk(dataset_dirs[0])
        if len(dataset_dirs) > 1:
            for dataset_dir in dataset_dirs[1:]:
                d = load_from_disk(dataset_dir)
                dataset = concatenate_datasets([dataset, d])
            dataset = dataset.shuffle()
    return dataset


def check_training_args(training_args):
    if training_args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            training_args.gradient_accumulation_steps))
    if training_args.train_batch_size % training_args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            training_args.gradient_accumulation_steps, training_args.train_batch_size))

    if training_args.phase1 and training_args.phase2:
        raise ValueError("Pre-training script has to be invoked for either phase 1 or phase 2, not both")


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

    # check arguments
    check_training_args(training_args)

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

    bert_config: BertConfig = None
    if model_args.config_file:
        bert_config = BertConfig.from_json_file(model_args.config_file)
    elif model_args.model_name:
        bert_config = get_bert_model_config(model_args.model_name)
    else:
        bert_config = get_bert_model_config("bert-tiny")

    logger.info(f"Instantiating a new config instance {bert_config}")

    # fresh MLM/NSP model
    model = BertForPreTraining(bert_config)

    if model_args.sparse_model:
        num_parameters = model.num_parameters()
        model = sparsify_model(model, model_args)
        logger.info(f"Model sparsified from {num_parameters} parameters, now has {model.num_parameters()} parameters")

    world_size = get_world_size(training_args)

    if world_size > 1:
        rank = torch.distributed.get_rank()
        dataset = load_dataset(data_args, rank)
        logger.info(f"Process with rank {rank} got assigned dataset shard of {len(dataset)} samples")
    else:
        dataset = load_dataset(data_args, 0)
        logger.info(f"Using a pre-training dataset of {len(dataset)} total samples")

    training_args.warmup_steps = int(training_args.max_steps * training_args.warmup_proportion)
    training_args.per_device_train_batch_size = training_args.per_device_train_batch_size // training_args.gradient_accumulation_steps

    # Training
    if training_args.phase1 and training_args.phase2:
        raise RuntimeError("Pre-training script has to be invoked for either phase 1 or phase 2, not both")

    logger.info(f"Starting {'phase 1' if training_args.phase1 else 'phase 2'}...")
    if training_args.checkpoint_dir and training_args.resume_from_checkpoint:
        logger.info(f"Pre-training continues from checkpoint {training_args.checkpoint_dir}")
        model = model.from_pretrained(training_args.checkpoint_dir)
        logger.info(f"Loaded model from checkpoint {training_args.checkpoint_dir}")
    elif training_args.phase2:
        model_state = torch.load(os.path.join(training_args.phase1_output_dir, "pytorch_model.bin"))
        model.load_state_dict(model_state)
        logger.info(f"Loaded model from phase 1, continue pre-training")

    trainer = BertTrainer(model=model, args=training_args, train_dataset=dataset,
                          optimizers=prepare_optimizer_and_scheduler(model, training_args))
    # checkpoint_dir is ignored if None
    trainer.train(model_path=training_args.checkpoint_dir)
    trainer.save_model()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
