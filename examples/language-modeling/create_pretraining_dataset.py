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
from datasets import load_from_disk, Dataset, load_dataset

from transformers import (BertTokenizer, PreTrainedTokenizer, HfArgumentParser)

nltk.download('punkt')
logger = logging.getLogger(__name__)


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join([x for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join([x for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


class TokenizerLambda(object):

    def __init__(self, tokenizer: PreTrainedTokenizer, text_column: str = "text"):
        self.tokenizer = tokenizer
        self.text_column: str = text_column

    def __call__(self, documents: List[str]) -> Dict[str, Any]:
        # batched version
        outputs = []
        for doc in documents[self.text_column]:
            tokenized_doc = []
            for sentence in doc:
                tokens = self.tokenizer.tokenize(sentence)
                tokenized_doc.append(tokens)
            outputs.append(tokenized_doc)
        return {'tokens': outputs}


def convert_instances_to_dataset(instances, tokenizer, max_seq_length, max_predictions_per_seq):
    """Create new HF dataset from `TrainingInstances."""

    features = collections.OrderedDict()
    num_instances = len(instances)
    features["input_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["input_mask"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["segment_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["masked_lm_positions"] = np.zeros([num_instances, max_predictions_per_seq], dtype="int32")
    features["masked_lm_ids"] = np.zeros([num_instances, max_predictions_per_seq], dtype="int32")
    features["next_sentence_labels"] = np.zeros(num_instances, dtype="int32")

    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0

        features["input_ids"][inst_index] = input_ids
        features["input_mask"][inst_index] = input_mask
        features["segment_ids"][inst_index] = segment_ids
        features["masked_lm_positions"][inst_index] = masked_lm_positions
        features["masked_lm_ids"][inst_index] = masked_lm_ids
        features["next_sentence_labels"][inst_index] = next_sentence_label

    return Dataset.from_dict(features)


def create_training_instances(all_documents, tokenizer: PreTrainedTokenizer, args):
    """Create `TrainingInstance`s from documents."""
    rng = random.Random(args.random_seed)
    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.get_vocab().keys())
    instances = []
    for _ in range(args.dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, args.max_seq_length, args.short_seq_prob,
                    args.masked_lm_prob, args.max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances


def create_instances_from_document(
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
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
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
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
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
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
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


def segment_sentences(docs):
    # batched version
    outputs = []
    for doc in docs["text"]:
        sentences = nltk.sent_tokenize(doc)
        if len(sentences) > 1:
            # remove category and references from the end of articles
            # TODO cleaning dataset is a better approach
            sentences.pop()
        outputs.append(sentences)
    return {'text': outputs}


@dataclass
class PreTrainingArguments:
    input_dataset: str = field(
        default="./small_wikipedia_sample",
        metadata={"help": "Input dataset name consisting of text docs used to create LM pre-training features"}
    )

    input_dataset_local: bool = field(
        default=True,
        metadata={"help": "If dataset is disk local, otherwise load dataset using load_dataset"}
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


def main():
    parser = HfArgumentParser((PreTrainingArguments))
    args, = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    if args.input_dataset_local:
        dataset = load_from_disk(args.input_dataset)
    else:
        dataset = load_dataset('wikipedia', "20200501.en", split='train')

    logger.info(f"Pre-training dataset has {len(dataset)} documents")
    logger.info(f"Segmenting pre-training dataset into sentences. Please wait...")
    dataset = dataset.map(segment_sentences, batched=True, remove_columns=dataset.column_names)
    logger.info(f"Segmented pre-training dataset into sentences, tokenizing sentences...")
    documents = dataset.map(TokenizerLambda(tokenizer), batched=True, remove_columns=dataset.column_names)

    all_documents = list([d for d in documents["tokens"]])
    logger.info(f"Creating training instances, please wait...")
    instances = create_training_instances(all_documents, tokenizer, args)
    show_samples = 3
    logger.info(f"Created {len(instances)} training instances. Here are a {show_samples} samples:")

    for idx, instance in enumerate(instances):
        if idx >= show_samples:
            break
        logger.info(f"Pre-training instance sample #{idx}:\n{str(instance)}")

    logger.info(f"Creating dataset for {len(instances)} training instances.")
    encoded_pre_training_dataset = convert_instances_to_dataset(instances, tokenizer, args.max_seq_length,
                                                                args.max_predictions_per_seq)

    encoded_pre_training_dataset.save_to_disk(args.output_dataset)
    logger.info(f"Prepared and saved pre-training dataset with {len(encoded_pre_training_dataset)} training instances.")


if __name__ == "__main__":
    main()
