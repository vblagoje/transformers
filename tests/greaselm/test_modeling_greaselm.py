# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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


import unittest
from copy import deepcopy

from huggingface_hub import hf_hub_download
from transformers import GreaseLMConfig, is_torch_available
from transformers.testing_utils import TestCasePlus, require_torch, slow, torch_device

from ..test_configuration_common import ConfigTester
from ..test_modeling_common import floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import GreaseLMForMultipleChoice, GreaseLMModel
    from transformers.models.greaselm.modeling_greaselm import (
        GREASELM_PRETRAINED_MODEL_ARCHIVE_LIST,
        GreaseLMEmbeddings,
        create_position_ids_from_input_ids,
    )


class GreaseLMModelTester:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 3
        self.seq_length = 128
        self.is_training = True
        self.use_input_mask = True
        self.use_token_type_ids = True
        self.use_labels = True
        self.vocab_size = 99
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.intermediate_size = 37
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 16
        self.type_sequence_label_size = 2
        self.initializer_range = 0.02
        self.num_labels = 3
        self.num_choices = 5
        self.scope = None

    def prepare_config_and_inputs(self, add_labels=False):
        input_ids = ids_tensor([self.batch_size, self.num_choices, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            attention_mask = random_attention_mask([self.batch_size, self.num_choices, self.seq_length])
            output_mask = random_attention_mask([self.batch_size, self.num_choices, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.num_choices, self.seq_length], self.type_vocab_size)

        n_node = 200
        concept_ids = torch.arange(end=n_node).repeat(self.batch_size, self.num_choices, 1).to(torch_device)
        adj_lengths = torch.zeros([self.batch_size, self.num_choices], dtype=torch.long).fill_(10).to(torch_device)

        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(torch_device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(torch_device)

        edge_index = [[edge_index] * self.num_choices] * self.batch_size
        edge_type = [[edge_type] * self.num_choices] * self.batch_size

        node_type_ids = torch.zeros([self.batch_size, self.num_choices, n_node], dtype=torch.long).to(torch_device)
        node_type_ids[:, :, 0] = 3
        node_scores = torch.zeros([self.batch_size, self.num_choices, n_node, 1]).to(torch_device)
        node_scores[:, :, 1] = 180

        special_nodes_mask = torch.zeros([self.batch_size, self.num_choices, n_node], dtype=torch.long).to(torch_device)

        config = self.get_config()
        if add_labels:
            return (
                config,
                input_ids,
                attention_mask,
                token_type_ids,
                torch.zeros([self.batch_size], dtype=torch.long).to(torch_device),
                output_mask,
                concept_ids,
                node_type_ids,
                node_scores,
                adj_lengths,
                special_nodes_mask,
                edge_index,
                edge_type,
            )
        else:
            return (
                config,
                input_ids,
                attention_mask,
                token_type_ids,
                output_mask,
                concept_ids,
                node_type_ids,
                node_scores,
                adj_lengths,
                special_nodes_mask,
                edge_index,
                edge_type,
            )

    def get_config(self):
        return GreaseLMConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        output_mask,
        concept_ids,
        node_type_ids,
        node_scores,
        adj_lengths,
        special_nodes_mask,
        edge_index,
        edge_type,
    ):
        concept_emb = hf_hub_download(repo_id="vblagoje/greaselm", filename="tzw.ent.npy")
        model = GreaseLMModel(config=config, pretrained_concept_emb_file=concept_emb)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask,
            token_type_ids,
            output_mask,
            concept_ids,
            node_type_ids,
            node_scores,
            adj_lengths,
            special_nodes_mask,
            edge_index,
            edge_type,
        )

    def create_and_check_for_multiple_choice(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        labels,
        special_tokens_mask,
        concept_ids,
        node_type_ids,
        node_scores,
        adj_lengths,
        special_nodes_mask,
        edge_index,
        edge_type,
    ):
        config.num_choices = self.num_choices
        concept_emb = hf_hub_download(repo_id="vblagoje/greaselm", filename="tzw.ent.npy")
        model = GreaseLMForMultipleChoice(config=config, pretrained_concept_emb_file=concept_emb)
        model.to(torch_device)
        model(
            input_ids,
            attention_mask,
            token_type_ids,
            labels,
            special_tokens_mask,
            concept_ids,
            node_type_ids,
            node_scores,
            adj_lengths,
            special_nodes_mask,
            edge_index,
            edge_type,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class GreaseLMModelTest(unittest.TestCase):

    all_model_classes = (
        (
            GreaseLMModel,
            GreaseLMForMultipleChoice,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = ()
    fx_compatible = False

    def setUp(self):
        self.model_tester = GreaseLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GreaseLMConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(add_labels=True)
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in GREASELM_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = GreaseLMModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class GreaseLMModelIntegrationTest(TestCasePlus):
    @slow
    def test_inference_no_head(self):
        model = GreaseLMModel.from_pretrained("vblagoje/greaselm")
