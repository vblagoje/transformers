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
Image/Text processor class for CLIP
"""
from typing import Any, Dict

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


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
        self.current_processor.start()

    def __call__(self, question_answer_example: Dict[str, Any], return_tensors=None, **kwargs):
        r"""
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
        """

        features = self.current_processor(question_answer_example)
        return BatchEncoding(data=dict(**features), tensor_type=return_tensors)

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
