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

        entailed_qa = [convert_qajson_to_entailment(e) for e in question_answer_example]
        qids, num_choices, lm_encoding = self.encode_question_answer_example(entailed_qa)
        assert num_choices > 0
        assert len(qids) == len(question_answer_example)

        # Load adj data
        features = self.current_processor(question_answer_example[0])

        kg_encoding: Dict[str, Any] = self.current_processor.load_sparse_adj_data_with_contextnode(
            features, self.max_node_num, [], num_choices
        )

        return KGEncoding(data={**lm_encoding, **kg_encoding})

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


class KGEncoding(UserDict):
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        super().__init__(data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def _to_device(self, obj, device):
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
            [`KGEncoding`]: The same instance after modification.
        """

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or isinstance(device, torch.device) or isinstance(device, int):
            self.data = {k: self._to_device(obj=v, device=device) for k, v in self.data.items()}
        else:
            raise TypeError(f"device must be a str, torch.device, or int, got {type(device)}")
        return self
