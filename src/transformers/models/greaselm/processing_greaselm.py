# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""
Processor class for GreaseLM
"""
from typing import Any, Dict, List

import torch

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils.logging import tqdm
from .convert_csqa import convert_qajson_to_entailment


class GreaseLMProcessor(ProcessorMixin):
    r"""
    Constructs a GreaseLM processor which wraps a GreaseLM feature extractor and a Roberta tokenizer into a single
    processor.

    [`GreaseLMProcessor`] offers all the functionalities of [`GreaseLMFeatureExtractor`] and [`RobertaTokenizerFast`].
    See the [`~GreaseLMProcessor.__call__`] and [`~GreaseLMProcessor.decode`] for more information.

    Args:
        feature_extractor ([`GreaseLMFeatureExtractor`]):
            The feature extractor is a required input.
        tokenizer ([`RobertaTokenizerFast`]):
            The tokenizer is a required input.
    """
    feature_extractor_class = "GreaseLMFeatureExtractor"
    tokenizer_class = ("RobertaTokenizer", "RobertaTokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self.max_node_num = 200
        self.max_seq_length = 512
        self.current_processor.start()

    def __call__(self, question_answer_example: List[Dict[str, Any]], return_tensors=None, **kwargs):
        r"""
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
        """

        entailed_qa = convert_qajson_to_entailment(question_answer_example[0])
        qids, labels, encoder_data, concepts_by_sents_list = self.encode_question_answer_example([entailed_qa])
        lm_encoding = dict(
            input_ids=encoder_data[0],
            attention_mask=encoder_data[1],
            token_type_ids=encoder_data[2],
            special_tokens_mask=encoder_data[3],
        )

        # Load adj data
        features = self.current_processor(question_answer_example[0])
        num_choices = encoder_data[0].size(1)
        assert num_choices > 0

        kg_encoding: Dict[str, Any] = self.current_processor.load_sparse_adj_data_with_contextnode(
            features, self.max_node_num, concepts_by_sents_list, num_choices
        )

        # add qids, labels, to batch encoding
        return BatchEncoding(data={**lm_encoding, **kg_encoding}, tensor_type=return_tensors)

    def encode_question_answer_example(self, entailed_qa_examples: List[Dict[str, Any]]):
        class InputExample(object):
            def __init__(self, example_id, question, contexts, endings, label=None):
                self.example_id = example_id
                self.question = question
                self.contexts = contexts
                self.endings = endings
                self.label = label

        class InputFeatures(object):
            def __init__(self, example_id, choices_features, label):
                self.example_id = example_id
                self.choices_features = [
                    {
                        "input_ids": input_ids,
                        "input_mask": input_mask,
                        "segment_ids": segment_ids,
                        "output_mask": output_mask,
                    }
                    for input_ids, input_mask, segment_ids, output_mask in choices_features
                ]
                self.label = label

        def read_examples(qa_entailed_statements: List[Dict[str, Any]]):
            examples = []
            for json_dic in qa_entailed_statements:
                label = ord(json_dic["answerKey"]) - ord("A") if "answerKey" in json_dic else 0
                contexts = json_dic["question"]["stem"]
                if "para" in json_dic:
                    contexts = json_dic["para"] + " " + contexts
                if "fact1" in json_dic:
                    contexts = json_dic["fact1"] + " " + contexts
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label,
                    )
                )

            return examples

        def simple_convert_examples_to_features(examples, label_list):
            """Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
            """
            label_map = {label: i for i, label in enumerate(label_list)}

            features = []
            concepts_by_sents_list = []
            for ex_index, example in tqdm(
                enumerate(examples), total=len(examples), desc="Converting examples to features"
            ):
                choices_features = []
                for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                    ans = example.question + " " + ending

                    encoded_input = self.tokenizer(
                        context,
                        ans,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_seq_length,
                        return_token_type_ids=True,
                        return_special_tokens_mask=True,
                    )
                    input_ids = encoded_input["input_ids"]
                    output_mask = encoded_input["special_tokens_mask"]
                    input_mask = encoded_input["attention_mask"]
                    segment_ids = encoded_input["token_type_ids"]

                    choices_features.append((input_ids, input_mask, segment_ids, output_mask))
                label = label_map[example.label]
                features.append(
                    InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label)
                )

            return features, concepts_by_sents_list

        def select_field(features, field):
            return [[choice[field] for choice in feature.choices_features] for feature in features]

        def convert_features_to_tensors(features):
            all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
            all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
            all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
            all_output_mask = torch.tensor(select_field(features, "output_mask"), dtype=torch.bool)
            all_label = torch.tensor([f.label for f in features], dtype=torch.long)
            return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

        examples = read_examples(entailed_qa_examples)
        features, concepts_by_sents_list = simple_convert_examples_to_features(
            examples, list(range(len(examples[0].endings)))
        )
        example_ids = [f.example_id for f in features]
        *data_tensors, all_label = convert_features_to_tensors(features)
        return example_ids, all_label, data_tensors, concepts_by_sents_list

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
