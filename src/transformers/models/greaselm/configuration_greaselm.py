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
""" greaselm configuration"""
from collections import OrderedDict
from typing import Mapping

from ...onnx import OnnxConfig
from ...utils import logging
from ..bert.configuration_bert import BertConfig


logger = logging.get_logger(__name__)

GREASELM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "vblagoje/greaselm": "https://huggingface.co/vblagoje/greaselm/resolve/main/config.json",
}


class GreaseLMConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a [`GreaseLMModel`] or a [`TFGreaseLMModel`]. It is
    used to instantiate a greaselm model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the greaselm
    [vblagoje/greaselm](https://huggingface.co/vblagoje/greaselm) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    The [`GreaseLMConfig`] class directly inherits [`BertConfig`]. It reuses the same defaults. Please check the parent
    class for more information.

    Args:
        num_gnn_layers (`int`, *optional*, defaults to 5):
            Number of GNN layers
        num_node_types (`int`, *optional*, defaults to 4):
            Number of node types in the graph
        num_edge_types (`int`, *optional*, defaults to 38):
            Number of edge types in the graph
        concept_dim (`int`, *optional*, defaults to 200):
            Dimension of the concept embeddings
        gnn_hidden_size (`int`, *optional*, defaults to 200):
            Hidden size of the GNN

    Examples:

    ```python
    >>> from transformers import GreaseLMConfig, GreaseLMModel

    >>> # Initializing a greaselm configuration
    >>> configuration = GreaseLMConfig()

    >>> # Initializing a model from the configuration
    >>> model = GreaseLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "greaselm"

    def __init__(
        self, num_gnn_layers=5, num_node_types=4, num_edge_types=38, concept_dim=200, gnn_hidden_size=200, **kwargs
    ):
        """Constructs GreaseLMConfig."""
        super().__init__(**kwargs)
        default_dropout = 0.2
        self.num_gnn_layers = num_gnn_layers
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.concept_dim = concept_dim
        self.gnn_hidden_size = gnn_hidden_size
        self.num_lm_gnn_attention_heads = kwargs.pop("num_lm_gnn_attention_heads", 2)
        self.fc_dim = kwargs.pop("fc_dim", 200)
        self.n_fc_layer = kwargs.pop("n_fc_layer", 0)
        self.p_emb = kwargs.pop("p_emb", default_dropout)
        self.p_gnn = kwargs.pop("p_gnn", default_dropout)
        self.p_fc = kwargs.pop("p_fc", default_dropout)
        self.ie_dim = kwargs.pop("ie_dim", 200)
        self.info_exchange = kwargs.pop("info_exchange", True)
        self.ie_layer_num = kwargs.pop("ie_layer_num", 1)
        self.sep_ie_layers = kwargs.pop("sep_ie_layers", False)
        self.layer_id = kwargs.pop("layer_id", -1)


class GreaseLMOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
