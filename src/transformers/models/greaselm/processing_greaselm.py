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
from collections import UserDict
from typing import Any, Dict, List, Optional, Union

import torch

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils.logging import tqdm
from .utils_greaselm import convert_commonsenseqa_to_entailment, convert_openbookqa_to_entailment


class GreaseLMProcessor(ProcessorMixin):
    r"""
    Constructs a GreaseLM processor which wraps a GreaseLM feature extractor and a Roberta tokenizer into a single
    processor.

    [`GreaseLMProcessor`] offers all the functionalities you need to prepare data for the model.

    It uses ['GreaseLMFeatureExtractor'] to convert CommonSenseQA or OpenBookQA question-answer example(s) into a
    batch of graph encodings and then encodes examples into a batch of language model encodings, finally combining
    graph and language model encodings into a model ready input.


    Args:
        feature_extractor ([`GreaseLMFeatureExtractor`]):
            The feature extractor is a required input.
        tokenizer ([`RobertaTokenizerFast`]):
            The tokenizer is a required input.
    """
    feature_extractor_class = "GreaseLMFeatureExtractor"
    tokenizer_class = ("RobertaTokenizer", "RobertaTokenizerFast")

    def __init__(self, feature_extractor, tokenizer, max_seq_length=128):
        super().__init__(feature_extractor, tokenizer)
        self.max_seq_length = max_seq_length
        self.converters = {
            "commonsenseqa": convert_commonsenseqa_to_entailment,
            "openbookqa": convert_openbookqa_to_entailment,
        }
        feature_extractor.start()

    def __call__(self, question_answer_example: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs):
        """

        Main method that takes question-answer example(s) and encodes them into a batch of language model encodings and
        a batch of graph encodings combining the two encodings ready for GreaseLM model input

        Args:
        question_answer_example ('Union[Dict[str, Any], List[Dict[str, Any]]]'):
            The input question answer example. It can be a single example or a list of examples
            See CommonSenseQA and/or OpenBookQA examples for more information

        Returns:
              [`dict`]: A [`dict`] with the following fields:

              - input_ids: (batch_size, num_choices, seq_len)
              - token_type_ids: (batch_size, num_choices, seq_len)
              - attention_mask: (batch_size, num_choices, seq_len)
              - special_tokens_mask: (batch_size, num_choices, seq_len)
              - labels: (batch_size,)
              - concept_ids: (batch_size, num_choices, max_node_num)
              - node_type_ids: (batch_size, num_choices, max_node_num)
              - node_scores: (batch_size, num_choices, max_node_num, 1)
              - adj_lengths: (batch_size,　num_choices)
              - special_nodes_mask: (batch_size, num_choices, max_node_num)
              - edge_index: list of size (batch_size, num_choices), where each entry is tensor[2, E]
              - edge_type: list of size (batch_size, num_choices), where each entry is tensor[E, ]
        """
        # Check for valid input
        if isinstance(question_answer_example, list):
            assert all([isinstance(e, dict) for e in question_answer_example])
        elif isinstance(question_answer_example, dict):
            # add batch dimension
            question_answer_example = [question_answer_example]
        else:
            raise ValueError("Input parameter 'question_answer_example' must be a "
                             f"Union[Dict[str, Any], List[Dict[str, Any]]] not {type(question_answer_example)}")

        converter = kwargs.get("question_answer_converter", None)
        if converter is None:
            # try known formats, i.e. commonsenseqa and openbookqa
            format_type = self.detect_type(question_answer_example[0])
            if format_type:
                converter = self.converters[format_type]
            else:
                raise ValueError(
                    f"Could not detect the dataset type of the input example. "
                    f"Currently supported datasets are {self.converters.keys()}. "
                    f"For new dataset examples use `question_answer_converter` argument."
                )
        # Step 1: convert QA examples to a list of entailed examples
        entailed_qa = [converter(e) for e in question_answer_example]

        # Step 2: encode entailed QA examples into a batch of LM encodings
        qids, num_choices, lm_encoding = self.encode_question_answer_example(entailed_qa)

        # Step 3: encode entailed QA examples into a batch of graph encodings
        kg_encoding = self.feature_extractor(question_answer_example, entailed_qa, num_choices)

        # Step 4: combine the two encodings ready for model input
        return LMKGEncoding(data={**lm_encoding, **kg_encoding})

    @staticmethod
    def detect_type(question_answer_example: Dict[str, Any]):
        if "question" in question_answer_example and "question_concept" in question_answer_example["question"]:
            return "commonsenseqa"
        elif "question" in question_answer_example and "stem" in question_answer_example["question"]:
            # most likely openbookqa
            return "openbookqa"
        else:
            return None

    def encode_question_answer_example(self, entailed_qa_examples: List[Dict[str, Any]]):
        class InputExample(object):
            def __init__(self, example_id, question, contexts, endings, label=None):
                self.example_id = example_id
                self.question = question
                self.contexts = contexts
                self.endings = endings
                self.label = label

        def read_examples(qa_entailed_statements: List[Dict[str, Any]]) -> List[InputExample]:
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

        def examples_to_features(examples: List[InputExample], label_list: List[int]) -> List[Dict[str, Any]]:
            label_map = {label: i for i, label in enumerate(label_list)}

            features = []
            for idx, example in tqdm(enumerate(examples), total=len(examples)):
                choices = []
                for context, ending in zip(example.contexts, example.endings):
                    ans = example.question + " " + ending
                    choices.append((context, ans))
                encoded_input = self.tokenizer(
                    choices,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_token_type_ids=True,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )
                label = label_map[example.label]
                features.append({"id": example.example_id, "choices": encoded_input, "label": label})

            return features

        examples = read_examples(entailed_qa_examples)
        features = examples_to_features(examples, list(range(len(examples[0].endings))))
        example_ids = [f["id"] for f in features]
        all_label = torch.tensor([f["label"] for f in features], dtype=torch.long)
        all_inputs_ids = torch.stack([f["choices"]["input_ids"] for f in features], dim=0)
        all_token_type_ids = torch.stack([f["choices"]["token_type_ids"] for f in features], dim=0)
        all_attention_mask = torch.stack([f["choices"]["attention_mask"] for f in features], dim=0)
        all_special_tokens_mask = torch.stack([f["choices"]["special_tokens_mask"] for f in features], dim=0)
        num_choices = all_inputs_ids.shape[1]  # second dim represents number of choices
        return (
            example_ids,
            num_choices,
            dict(
                input_ids=all_inputs_ids,
                token_type_ids=all_token_type_ids,
                attention_mask=all_attention_mask,
                special_tokens_mask=all_special_tokens_mask,
                labels=all_label,
            ),
        )

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


# It would be great to use BatchEncoding here.
# However, BatchEncoding doesn't support moving unequal length lists of tensors to device
class LMKGEncoding(UserDict):
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        super().__init__(data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def _to_device(self, obj: Any, device):
        """
        Recursively peels of lists and tuples to tensors and moves them to the given device.
        Note: we can't stack KG lists of tensors as they are of different sizes.

        :param obj: The object to be moved to the device
        :param device: The device to put the model on
        :return: A list of the items in the tuple or list that have been moved to the device.
        """
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)

    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        """
        Args:
        Send all values to device by calling `v.to(device)` (PyTorch only).
            device (`str` or `torch.device`): The device to put the tensors on.
        Returns:
            [`LMKGEncoding`]: The same instance after modification.
        """

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or isinstance(device, torch.device) or isinstance(device, int):
            self.data = {k: self._to_device(obj=v, device=device) for k, v in self.data.items()}
        else:
            raise TypeError(f"device must be a str, torch.device, or int, got {type(device)}")
        return self
