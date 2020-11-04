# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
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

"""Create masked LM/next sentence masked_lm dataset for BERT."""
import collections
import logging
import random
from typing import List, Dict, Any, Optional

import nltk
import numpy as np
from dataclasses import dataclass, field
from datasets import load_from_disk, load_dataset, Sequence, Features, Value

from transformers import (PreTrainedTokenizer, HfArgumentParser, BertTokenizerFast)

nltk.download('punkt')
logger = logging.getLogger(__name__)


@dataclass
class PreTrainingArguments:
    input_dataset: str = field(
        default="./small_wikipedia_sample",
        metadata={"help": "Input dataset name consisting of text docs used to create LM pre-training features"}
    )

    input_dataset_remote: bool = field(
        default=False,
        metadata={"help": "If dataset is remote, get it using load_dataset, otherwise load dataset using load_dataset"}
    )

    output_dataset: str = field(
        default="./pre_training_dataset",
        metadata={"help": "Output dataset name with created LM pre-training features"}
    )

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after WordPiece tokenization. \n"
                          "Sequences longer than this will be truncated, and sequences shorter \n"
                          "than this will be padded."}
    )

    dupe_factor: Optional[int] = field(
        default=2,
        metadata={"help": "Number of times to duplicate the input data (with different masks)."}
    )

    num_proc: Optional[int] = field(
        default=0,
        metadata={"help": "Number of processes for multiprocessing. By default it doesn't use multiprocessing."}
    )

    num_shards: Optional[int] = field(
        default=10,
        metadata={"help": "Number of shards applied to output dataset"}
    )

    max_predictions_per_seq: Optional[int] = field(
        default=20,
        metadata={"help": "The maximum number of masked tokens per sequence"}
    )

    random_seed: Optional[int] = field(
        default=12345,
        metadata={"help": "Random seed initialization"}
    )

    masked_lm_prob: Optional[float] = field(
        default=0.15,
        metadata={"help": "The masked LM probability"}
    )

    short_seq_prob: Optional[float] = field(
        default=0.1,
        metadata={"help": "Probability to create a sequence shorter than maximum sequence length"}
    )

    tokenizer: Optional[str] = field(
        default='bert-base-uncased',
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


class TokenizerLambda(object):

    def __init__(self, tokenizer: PreTrainedTokenizer, text_column: str = "text"):
        self.tokenizer = tokenizer
        self.text_column: str = text_column

    def __call__(self, documents: List[str]) -> Dict[str, Any]:
        # batched version
        outputs = []
        for doc in documents[self.text_column]:
            doc_encoded = self.tokenizer.batch_encode_plus(doc, add_special_tokens=False)
            outputs.append(doc_encoded["input_ids"])
        return {'tokens': outputs}


class TrainingInstanceFactoryLambda(object):

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int,
                 short_seq_prob: float, masked_lm_prob: float, max_predictions_per_seq: int, dupe_factor: int,
                 text_column: str = "tokens"):
        self.tokenizer = tokenizer
        # for some reason @dataclass instances are not pickable
        # would use args as a constructor parameter
        # self.args = args
        self.max_seq_length = max_seq_length
        self.short_seq_prob = short_seq_prob
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.dupe_factor = dupe_factor
        self.text_column: str = text_column
        self.MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                                       ["index", "label"])

    def __call__(self, documents: List[str]) -> Dict[str, Any]:
        """Create `TrainingInstance`s from documents."""
        # rng = random.Random(self.args.random_seed)
        rng = random.Random(12345)
        # Remove empty documents
        # all_documents = [x for x in all_documents if x]
        # rng.shuffle(all_documents)
        # print(f"len(documents): {len(documents[self.text_column])}")
        vocab_words = list(self.tokenizer.get_vocab().keys())
        all_tokens, all_segment_ids, all_is_random_next, all_masked_lm_positions, all_masked_lm_labels = [], [], [], [], []

        for _ in range(self.dupe_factor):
            for document_index in range(len(documents[self.text_column])):
                tokens, segment_ids, is_random_next, masked_lm_positions, masked_lm_labels = self.create_instances_from_document(
                    documents[self.text_column], document_index, self.max_seq_length, self.short_seq_prob,
                    self.masked_lm_prob, self.max_predictions_per_seq, vocab_words, rng)

                all_tokens.extend(tokens)
                all_segment_ids.extend(segment_ids)
                all_is_random_next.extend(is_random_next)
                all_masked_lm_positions.extend(masked_lm_positions)
                all_masked_lm_labels.extend(masked_lm_labels)

        return {"input_tokens": all_tokens, "segment_ids": all_segment_ids,
                "is_random_next": all_is_random_next,
                "masked_lm_positions": all_masked_lm_positions,
                "masked_lm_labels": all_masked_lm_labels}

    def create_instances_from_document(self,
                                       all_documents, document_index, max_seq_length, short_seq_prob,
                                       masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
        """Creates `TrainingInstance`s for a single document."""
        document = all_documents[document_index]

        # Account for [CLS], [SEP], [SEP]
        max_num_tokens = max_seq_length - 3

        # We *usually* want to fill up the entire sequence since we are padding
        # to `max_seq_length` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pre-training and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `max_seq_length` is a hard limit.
        target_seq_length = max_num_tokens
        if rng.random() < short_seq_prob:
            target_seq_length = rng.randint(2, max_num_tokens)

        # We DON'T just concatenate all of the tokens from a document into a long
        # sequence and choose an arbitrary split point because this would make the
        # next sentence prediction task too easy. Instead, we split the input into
        # segments "A" and "B" based on the actual "sentences" provided by the user
        # input.
        current_chunk = []
        current_length = 0
        i = 0
        all_tokens, all_segment_ids, all_is_random_next, all_masked_lm_positions, all_masked_lm_labels = [], [], [], [], []
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
                        a_end = rng.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    # Random next
                    is_random_next = False
                    if len(current_chunk) == 1 or rng.random() < 0.5:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        random_document_index = None
                        for _ in range(10):
                            random_document_index = rng.randint(0, len(all_documents) - 1)
                            if random_document_index != document_index:
                                break

                        # If picked random document is the same as the current document
                        if random_document_index == document_index:
                            is_random_next = False

                        random_document = all_documents[random_document_index]
                        random_start = rng.randint(0, len(random_document) - 1)
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
                    self.truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    tokens = []
                    segment_ids = []
                    tokens.append(self.tokenizer.cls_token_id)
                    segment_ids.append(0)
                    for token in tokens_a:
                        tokens.append(token)
                        segment_ids.append(0)

                    tokens.append(self.tokenizer.sep_token_id)
                    segment_ids.append(0)

                    for token in tokens_b:
                        tokens.append(token)
                        segment_ids.append(1)
                    tokens.append(self.tokenizer.sep_token_id)
                    segment_ids.append(1)

                    (tokens, masked_lm_positions,
                     masked_lm_labels) = self.create_masked_lm_predictions(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)

                    all_tokens.append(tokens)
                    all_segment_ids.append(segment_ids)
                    all_is_random_next.append(is_random_next)
                    all_masked_lm_positions.append(masked_lm_positions)
                    all_masked_lm_labels.append(masked_lm_labels)
                current_chunk = []
                current_length = 0
            i += 1

        return all_tokens, all_segment_ids, all_is_random_next, all_masked_lm_positions, all_masked_lm_labels

    def create_masked_lm_predictions(self, tokens, masked_lm_prob,
                                     max_predictions_per_seq, vocab_words, rng):
        """Creates the predictions for the masked LM objective."""

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == self.tokenizer.cls_token_id or token == self.tokenizer.sep_token_id:
                continue
            cand_indexes.append(i)

        rng.shuffle(cand_indexes)

        output_tokens = list(tokens)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = self.tokenizer.mask_token_id
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    random_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
                    masked_token = self.tokenizer.convert_tokens_to_ids(random_token)

            output_tokens[index] = masked_token
            masked_lms.append(self.MaskedLmInstance(index=index, label=tokens[index]))

        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)

        return output_tokens, masked_lm_positions, masked_lm_labels

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens, rng):
        """Truncates a pair of sequences to a maximum sequence length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break

            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if rng.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()


class InstanceConverterLambda(object):

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int, max_predictions_per_seq: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_predictions_per_seq = max_predictions_per_seq

    def __call__(self, documents: List[str]) -> Dict[str, Any]:
        return self.convert_instances_to_dataset(documents)

    def convert_instances_to_dataset(self, instances):
        """Create new HF dataset from training instances"""

        num_instances = len(instances["input_tokens"])
        all_input_ids = np.zeros([num_instances, self.max_seq_length], dtype="int32")
        all_input_mask = np.zeros([num_instances, self.max_seq_length], dtype="int8")
        all_segment_ids = np.zeros([num_instances, self.max_seq_length], dtype="int8")
        all_masked_labels = np.full([num_instances, self.max_seq_length], fill_value=-100, dtype="int32")
        all_next_sentence_labels = np.zeros(num_instances, dtype="int8")

        for idx, (tokens, segment_ids, masked_lm_positions, masked_lm_labels, random_next) in enumerate(
                zip(instances['input_tokens'],
                    instances['segment_ids'],
                    instances['masked_lm_positions'],
                    instances['masked_lm_labels'],
                    instances['is_random_next'])):
            input_ids = tokens
            input_mask = [1] * len(input_ids)
            masked_labels = np.full([self.max_seq_length, ], -100, dtype="int32")
            masked_labels[masked_lm_positions] = masked_lm_labels

            # masked_lm_weights omitted for pytorch, needed for tf features
            all_input_ids[idx][:len(tokens)] = tokens
            all_input_mask[idx][:len(tokens)] = input_mask
            all_segment_ids[idx][:len(tokens)] = segment_ids
            all_masked_labels[idx] = masked_labels
            all_next_sentence_labels[idx] = 1 if random_next else 0

        return {"input_ids": all_input_ids,
                "attention_mask": all_input_mask,
                "token_type_ids": all_segment_ids,
                "labels": all_masked_labels,
                "next_sentence_label": all_next_sentence_labels}


def segment_sentences(docs):
    # batched version
    outputs = []
    for doc in docs["text"]:
        sentences = nltk.sent_tokenize(doc)
        if len(sentences) > 1:
            # remove category and references from the end of wikipedia document
            # TODO cleaning dataset is a better approach
            sentences.pop()
        outputs.append(sentences)
    return {'text': outputs}


def main():
    parser = HfArgumentParser((PreTrainingArguments))
    args, = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)

    if not args.input_dataset_remote:
        dataset = load_from_disk(args.input_dataset)
    else:
        dataset = load_dataset('wikipedia', "20200501.en", split='train')

    logger.info(f"Pre-training dataset has {len(dataset)} documents")
    logger.info(f"Segmenting pre-training dataset into sentences. Please wait...")
    dataset = dataset.map(segment_sentences, batched=True, remove_columns=dataset.column_names,
                          num_proc=args.num_proc if args.num_proc > 0 else None)

    logger.info(f"Segmented pre-training dataset into sentences, tokenizing sentences...")
    documents = dataset.map(TokenizerLambda(tokenizer), batched=True, remove_columns=dataset.column_names,
                            num_proc=args.num_proc if args.num_proc > 0 else None)

    logger.info(f"Creating training instances using {len(documents)} documents, please wait...")
    instances = documents.map(TrainingInstanceFactoryLambda(tokenizer,
                                                            max_seq_length=args.max_seq_length,
                                                            short_seq_prob=args.short_seq_prob,
                                                            masked_lm_prob=args.masked_lm_prob,
                                                            dupe_factor=args.dupe_factor,
                                                            max_predictions_per_seq=args.max_predictions_per_seq),
                              batched=True, remove_columns=documents.column_names,
                              num_proc=args.num_proc if args.num_proc > 0 else None)

    show_samples = 3
    logger.info(f"Created {len(instances)} training instances. Here are a {show_samples} samples:")

    for idx in range(show_samples):
        logger.info(f"Pre-training instance sample #{idx}:"
                    f"\ntokens:{tokenizer.decode(instances[idx]['input_tokens'])} "
                    f"\nsegment_ids:{str(instances[idx]['segment_ids'])} "
                    f"\nis_random_next:{str(instances[idx]['is_random_next'])} "
                    f"\nmasked_lm_positions:{str(instances[idx]['masked_lm_positions'])} "
                    f"\nmasked_lm_labels:{tokenizer.decode(instances[idx]['masked_lm_labels'])}\n")

    logger.info(f"Creating dataset of pre-training instances.")
    f = Features({'input_ids': Sequence(feature=Value(dtype='int32')),
                  'attention_mask': Sequence(feature=Value(dtype='int8')),
                  'token_type_ids': Sequence(feature=Value(dtype='int8')),
                  'labels': Sequence(feature=Value(dtype='int32')),
                  'next_sentence_label': Value(dtype='int8'),
                  })

    for shard_i in range(args.num_shards):
        instances_shard = instances.shard(num_shards=args.num_shards, index=shard_i)
        logger.info(f"Processing shard {shard_i} with {len(instances_shard)} training instances.")
        pre_training_dataset = instances_shard.map(
            InstanceConverterLambda(tokenizer, args.max_seq_length, args.max_predictions_per_seq),
            features=f, batched=True, remove_columns=instances.column_names,
            num_proc=args.num_proc if args.num_proc > 0 else None)
        shard_output_file = "_".join([args.output_dataset, str(shard_i)])
        logger.info(f"Saving sharded dataset {shard_output_file} to disk.")
        pre_training_dataset.save_to_disk(shard_output_file)
    logger.info(f"Completed!")


if __name__ == "__main__":
    main()
