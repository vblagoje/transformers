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
import logging
import random
from typing import Dict, List, Optional, Any

import nltk
from dataclasses import dataclass, field
from datasets import load_dataset

from transformers import (
    HfArgumentParser,
    PreTrainedTokenizer,
    BertTokenizerFast, )

nltk.download('punkt')

logger = logging.getLogger(__name__)


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
    tokenizer_name: Optional[str] = field(
        default='bert-base-uncased',
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


def main():
    parser = HfArgumentParser((ModelArguments))
    model_args, = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Fast tokenizer helps a lot with the speed of encoding
    tokenizer = BertTokenizerFast.from_pretrained(model_args.tokenizer_name)

    bert_dataset = load_dataset('wikipedia', "20200501.en", split='train')
    nsp = AndBNextSentencePredictionProcessor(tokenizer=tokenizer)
    logger.info(f"Encoding bert dataset containing {len(bert_dataset)} articles")

    encoded_bert_dataset = bert_dataset.map(nsp, batched=True, remove_columns=bert_dataset.column_names)
    encoded_bert_dataset.save_to_disk("./bert_encoded_all")


if __name__ == "__main__":
    main()
