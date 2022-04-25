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
"""PyTorch greaselm model."""

import math
import os
from os.path import exists as file_exists
from typing import List, Optional, Tuple, Union

import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.autograd import Variable

from huggingface_hub import hf_hub_download
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter

from ...activations import ACT2FN, gelu
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, MultipleChoiceModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_greaselm import GreaseLMConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "vblagoje/greaselm"
_CONFIG_FOR_DOC = "GreaseLMConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

GREASELM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "vblagoje/greaselm",
    # See all greaselm models at https://huggingface.co/models?filter=greaselm
]

_CONCEPT_EMBEDDINGS_FILE_NAME = "tzw.ent.npy"


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def make_one_hot(labels, C):
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable. labels : torch.autograd.Variable of
    torch.cuda.LongTensor
        (N, ), where N is batch size. Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    """
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


class GATConvE(MessagePassing):
    """
    Args:
        emb_dim (int): dimensionality of GNN hidden states
        n_ntype (int): number of node types (e.g. 4)
        n_etype (int): number of edge relation types (e.g. 38)
    """

    def __init__(self, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        # For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3 * emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3 * emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2 * emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        # For final MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x, edge_index, edge_type, node_type, node_feature_extra, return_attention_weights=False):
        """
        x: [N, emb_dim] edge_index: [2, E] edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39] node_type
        [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8] node_feature_extra [N, dim]
        """

        # Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype + 1)  # [E, 39]
        self_edge_vec = torch.zeros(x.size(0), self.n_etype + 1).to(edge_vec.device)
        self_edge_vec[:, self.n_etype] = 1

        head_type = node_type[edge_index[0]]  # [E,] #head=src
        tail_type = node_type[edge_index[1]]  # [E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype)  # [E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype)  # [E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1)  # [E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype)  # [N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1)  # [N,8]

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0)  # [E+N, ?]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0)  # [E+N, ?]
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1))  # [E+N, emb_dim]

        # Add self loops to edge_index
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  # [2, E+N]

        x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)  # [N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, edge_index, x_i, x_j, edge_attr):  # i: tgt, j:src
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2 * self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(
            -1, self.head_count, self.dim_per_head
        )  # [E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(
            -1, self.head_count, self.dim_per_head
        )  # [E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]

        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2)  # [E, heads]
        src_node_index = edge_index[0]  # [E,]
        alpha = softmax(scores, src_node_index)  # [E, heads] #group by src side node
        self._alpha = alpha

        # adjust by outgoing degree of src
        E = edge_index.size(1)  # n_edges
        N = int(src_node_index.max()) + 1  # n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce="sum")[src_node_index]  # [E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1)  # [E, heads]

        out = msg * alpha.view(-1, self.head_count, 1)  # [E, heads, _dim]
        return out.view(-1, self.head_count * self.dim_per_head)  # [E, emb_dim]


class CustomizedEmbedding(nn.Module):
    def __init__(
        self,
        concept_num,
        concept_in_dim,
        concept_out_dim,
        use_contextualized=False,
        pretrained_concept_emb=None,
        freeze_ent_emb=True,
        scale=1.0,
        init_range=0.02,
    ):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        if not use_contextualized:
            self.emb = nn.Embedding(concept_num + 2, concept_in_dim)
            if pretrained_concept_emb is not None:
                self.emb.weight.data.fill_(0)
                self.emb.weight.data[:concept_num].copy_(pretrained_concept_emb)
            else:
                self.emb.weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                freeze_net(self.emb)

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = ACT2FN["gelu_new"]

    def forward(self, index, contextualized_emb=None):
        """
        index: size (bz, a) contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if contextualized_emb is not None:
            assert index.size(0) == contextualized_emb.size(0)
            if hasattr(self, "cpt_transform"):
                contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
            else:
                contextualized_emb = contextualized_emb * self.scale
            emb_dim = contextualized_emb.size(-1)
            return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        else:
            if hasattr(self, "cpt_transform"):
                return self.activation(self.cpt_transform(self.emb(index) * self.scale))
            else:
                return self.emb(index) * self.scale


class MatrixVectorScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k) k: tensor of shape (n*b, l, d_k) v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn


class MultiheadAttPoolLayer(nn.Module):
    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original) k: tensor of shape (b, l, d_k_original) mask: tensor of shape (b, l)
        (optional, default None) returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters ---------- num_layers: number of hidden layers
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        batch_norm=False,
        init_last_layer_bias_to_zero=False,
        layer_norm=False,
        activation="gelu_new",
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f"{i}-Linear", nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f"{i}-Dropout", nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f"{i}-BatchNorm1d", nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f"{i}-LayerNorm", nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f"{i}-{activation}", ACT2FN[activation.lower()])
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


class GreaseLMEncoder(nn.Module):
    def __init__(self, config, dropout=0.2):
        super().__init__()
        self.config = config
        self.k = config.k
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(config.n_etype + 1 + config.n_ntype * 2, config.gnn_hidden_size),
            torch.nn.BatchNorm1d(config.gnn_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.gnn_hidden_size, config.gnn_hidden_size),
        )

        self.layer = nn.ModuleList([GreaseLMLayer(config) for _ in range(config.num_hidden_layers)])
        self.gnn_layers = nn.ModuleList(
            [
                GATConvE(config.gnn_hidden_size, config.n_ntype, config.n_etype, self.edge_encoder)
                for _ in range(config.k)
            ]
        )
        self.activation = ACT2FN["gelu_new"]
        self.dropout_rate = dropout

        self.sep_ie_layers = config.sep_ie_layers
        if self.sep_ie_layers:
            self.ie_layers = nn.ModuleList(
                [
                    MLP(
                        config.hidden_size + config.concept_dim,
                        config.ie_dim,
                        config.hidden_size + config.concept_dim,
                        config.ie_layer_num,
                        config.p_fc,
                    )
                    for _ in range(config.k)
                ]
            )
        else:
            ie_layer_size = config.hidden_size + config.concept_dim
            self.ie_layer = MLP(ie_layer_size, config.ie_dim, ie_layer_size, config.ie_layer_num, config.p_fc)

        self.num_hidden_layers = config.num_hidden_layers
        self.info_exchange = config.info_exchange

    def forward(
        self,
        hidden_states,
        attention_mask,
        special_tokens_mask,
        head_mask,
        _X,
        edge_index,
        edge_type,
        _node_type,
        _node_feature_extra,
        special_nodes_mask,
        output_attentions=False,
        output_hidden_states=True,
    ):

        """
        :param hidden_states:
              (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, sent_dim)`):
        :param attention_mask:
              (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 1, 1, seq_len)`):
        :param special_tokens_mask:
              (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, seq_len)`):
                   Token type ids for the language model.
        :param head_mask: list of shape [num_hidden_layers]
        :param _X:
             (:obj:`torch.FloatTensor` of shape :obj:`(total_n_nodes, node_dim)`):
               `total_n_nodes` = batch_size * num_nodes
        :param edge_index:
              (:obj:`torch.LongTensor` of shape :obj:`(2, E)`):
        :param edge_type:
              (:obj:`torch.LongTensor` of shape :obj:`(E, )`):
        :param _node_type:
              (:obj:`torch.LongTensor` of shape :obj:`(total_n_nodes,)`):
        :param _node_feature_extra:
             (:obj:`torch.FloatTensor` of shape :obj:`(total_n_nodes, node_dim)`):
               `total_n_nodes` = batch_size * num_nodes
        :param special_nodes_mask:
              (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, max_node_num)`):
        :param output_attentions: (:obj:`bool`) Whether or not to return the attentions tensor.
        :param output_hidden_states: (:obj:`bool`) Whether or not to return the hidden states tensor.
        """
        bs = hidden_states.size(0)
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            # LM
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            if i >= self.num_hidden_layers - self.k:
                # GNN
                gnn_layer_index = i - self.num_hidden_layers + self.k
                _X = self.gnn_layers[gnn_layer_index](_X, edge_index, edge_type, _node_type, _node_feature_extra)
                _X = self.activation(_X)
                _X = F.dropout(_X, self.dropout_rate, training=self.training)

                # Exchange info between LM and GNN hidden states (Modality interaction)
                if self.info_exchange == True or (
                    self.info_exchange == "every-other-layer" and (i - self.num_hidden_layers + self.k) % 2 == 0
                ):
                    X = _X.view(bs, -1, _X.size(1))  # [bs, max_num_nodes, node_dim]
                    context_node_lm_feats = hidden_states[:, 0, :]  # [bs, sent_dim]
                    context_node_gnn_feats = X[:, 0, :]  # [bs, node_dim]
                    context_node_feats = torch.cat([context_node_lm_feats, context_node_gnn_feats], dim=1)
                    if self.sep_ie_layers:
                        context_node_feats = self.ie_layers[gnn_layer_index](context_node_feats)
                    else:
                        context_node_feats = self.ie_layer(context_node_feats)
                    context_node_lm_feats, context_node_gnn_feats = torch.split(
                        context_node_feats, [context_node_lm_feats.size(1), context_node_gnn_feats.size(1)], dim=1
                    )
                    hidden_states[:, 0, :] = context_node_lm_feats
                    X[:, 0, :] = context_node_gnn_feats
                    _X = X.view_as(_X)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs, _X  # last-layer hidden state, (all hidden states), (all attentions)
        # return BaseModelOutputWithPastAndCrossAttentions(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_decoder_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attentions,
        #     cross_attentions=all_cross_attentions,
        # )


# Copied from transformers.models.roberta.modeling_roberta.RobertaEmbeddings with Roberta->GreaseLM
class GreaseLMEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->GreaseLM
class GreaseLMSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in GreaseLMModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class GreaseLMSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->GreaseLM
class GreaseLMAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = GreaseLMSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = GreaseLMSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class GreaseLMIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class GreaseLMOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->GreaseLM
class GreaseLMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = GreaseLMAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = GreaseLMAttention(config, position_embedding_type="absolute")
        self.intermediate = GreaseLMIntermediate(config)
        self.output = GreaseLMOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertPooler
class GreaseLMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.roberta.modeling_roberta.RobertaPreTrainedModel with Roberta->GreaseLM,roberta->greaselm
class GreaseLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GreaseLMConfig
    base_model_prefix = "greaselm"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GreaseLMEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


GREASELM_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GreaseLMConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GREASELM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`RobertaTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare greaselm Model transformer outputting raw hidden-states without any specific head on top.",
    GREASELM_START_DOCSTRING,
)
class GreaseLMModel(GreaseLMPreTrainedModel):
    """
    Answering complex questions about textual narratives requires reasoning over both stated context and the world
    knowledge that underlies it. However, pretrained language models (LM), the foundation of most modern QA systems, do
    not robustly represent latent relationships between concepts, which is necessary for reasoning. While knowledge
    graphs (KG) are often used to augment LMs with structured representations of world knowledge, it remains an open
    question how to effectively fuse and reason over the KG representations and the language context, which provides
    situational constraints and nuances. In this work, we propose GreaseLM, a new model that fuses encoded
    representations from pretrained LMs and graph neural networks over multiple layers of modality interaction
    operations. Information from both modalities propagates to the other, allowing language context representations to
    be grounded by structured world knowledge, and allowing linguistic nuances (e.g., negation, hedging) in the context
    to inform the graph representations of knowledge.

    Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang, Christopher D. Manning, Jure Leskovec


    For more details see: https://arxiv.org/abs/2201.08860

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(
        self, config, pretrained_concept_emb_file=None, freeze_ent_emb=True, add_pooling_layer=True, dropout=0.2
    ):
        super().__init__(config)
        self.config = config
        assert pretrained_concept_emb_file is not None and file_exists(
            pretrained_concept_emb_file
        ), f"Pretrained concept embedding file not found: {pretrained_concept_emb_file}"

        pretrained_concept_emb = torch.tensor(np.load(pretrained_concept_emb_file), dtype=torch.float)
        concept_num, concept_in_dim = pretrained_concept_emb.size(0), pretrained_concept_emb.size(1)
        self.hidden_size = config.concept_dim
        self.emb_node_type = nn.Linear(config.n_ntype, config.concept_dim // 2)

        self.basis_f = "sin"  # ['id', 'linact', 'sin', 'none']
        if self.basis_f in ["id"]:
            self.emb_score = nn.Linear(1, config.concept_dim // 2)
        elif self.basis_f in ["linact"]:
            self.B_lin = nn.Linear(1, config.concept_dim // 2)
            self.emb_score = nn.Linear(config.concept_dim // 2, config.concept_dim // 2)
        elif self.basis_f in ["sin"]:
            self.emb_score = nn.Linear(config.concept_dim // 2, config.concept_dim // 2)

        self.Vh = nn.Linear(config.concept_dim, config.concept_dim)
        self.Vx = nn.Linear(config.concept_dim, config.concept_dim)

        self.activation = ACT2FN["gelu_new"]
        self.dropout = nn.Dropout(dropout)
        self.dropout_e = nn.Dropout(dropout)
        self.cpnet_vocab_size = config.n_concept
        self.concept_emb = (
            CustomizedEmbedding(
                concept_num=config.n_concept,
                concept_out_dim=config.concept_dim,
                use_contextualized=False,
                concept_in_dim=config.concept_in_dim,
                pretrained_concept_emb=pretrained_concept_emb,
                freeze_ent_emb=freeze_ent_emb,
            )
            if config.k >= 0
            else None
        )
        self.embeddings = GreaseLMEmbeddings(config)
        self.encoder = GreaseLMEncoder(config)
        self.pooler = GreaseLMPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        if os.path.isdir(pretrained_model_name_or_path):
            concept_emb = os.path.join(pretrained_model_name_or_path, _CONCEPT_EMBEDDINGS_FILE_NAME)
        else:
            concept_emb = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=_CONCEPT_EMBEDDINGS_FILE_NAME, **kwargs
            )
        kwargs["pretrained_concept_emb_file"] = concept_emb
        return super(GreaseLMModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        """
        edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E) edge_type_init: list of (n_examples,
        ). each entry is torch.tensor(E, )
        """

        def flatten(iterable):
            for item in iterable:
                if isinstance(item, list):
                    yield from flatten(item)
                else:
                    yield item

        edge_index_init = list(flatten(edge_index_init))
        edge_type_init = list(flatten(edge_type_init))
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1)  # [2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0)  # [total_E,]
        return edge_index, edge_type

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        special_tokens_mask,
        concept_ids,
        node_type_ids,
        node_scores,
        adj_lengths,
        special_nodes_mask,
        edge_index,
        edge_type,
        position_ids=None,
        head_mask=None,
        emb_data=None,
        cache_output=False,
        output_hidden_states=True,
    ):
        """
        :param input_ids:
              (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_len)`):
                   Input ids for the language model.
        :param attention_mask:
              (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_len)`):
                   Attention mask for the language model.
        :param token_type_ids:
              (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_len)`):
                   Token type ids for the language model.
        :param special_tokens_mask:
              (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, seq_len)`):
                   Output mask for the language model.
        :param concept_ids:
            (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, max_node_num)`):
                 Resolved conceptnet ids.
        :param node_type_ids:
              (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_nodes)`):
                   0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        :param node_scores:
               (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, number_of_choices, max_node_num, 1)`):
                 LM relevancy scores for each resolved conceptnet id
        :param adj_lengths:
            (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices)`):
                 Adjacency matrix lengths for each batch sample.
        :param special_nodes_mask:
              (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, number_of_nodes)`):
                 Mask identifying special nodes in the graph (interaction node in the GreaseLM paper).
        :param edge_index:
             torch.tensor(2, E)) where E is the total number of edges in the particular graph.
        :param edge_type:
             torch.tensor(E, ) where E is the total number of edges in the particular graph.
        :param position_ids:
             (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_len)`, `optional`, defaults to :obj:`None`):
                 Indices of positions of each input sequence tokens in the position embeddings.
        :param head_mask:
                 list of shape [num_hidden_layers]
        :param emb_data:
             torch.tensor(batch_size, number_of_choices, max_node_num, emb_dim)
        :param cache_output:
                 Whether to cache the output of the language model.
        :param output_hidden_states: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                 If set to ``True``, the model will return all hidden-states.

        """

        def merge_first_two_dim(x):
            return x.view(-1, *(x.size()[2:]))

        (
            input_ids,
            attention_mask,
            token_type_ids,
            special_tokens_mask,
            concept_ids,
            node_type_ids,
            node_scores,
            adj_lengths,
            special_nodes_mask,
        ) = [
            merge_first_two_dim(t)
            for t in [
                input_ids,
                attention_mask,
                token_type_ids,
                special_tokens_mask,
                concept_ids,
                node_type_ids,
                node_scores,
                adj_lengths,
                special_nodes_mask,
            ]
        ]
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        node_scores = torch.zeros_like(node_scores)

        # LM inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 1D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if len(attention_mask.size()) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif len(attention_mask.size()) == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise ValueError("Attnetion mask should be either 1D or 2D.")

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        # GNN inputs
        concept_ids[concept_ids == 0] = self.cpnet_vocab_size + 2
        gnn_input = self.concept_emb(concept_ids - 1, emb_data).to(node_scores.device)
        gnn_input[:, 0] = 0
        # H - node features from the previous layer
        H = self.dropout_e(gnn_input)  # (batch_size, n_node, dim_node)

        # Normalize node sore (use norm from Z)
        _mask = (
            torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)
        ).float()  # 0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :]  # [batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2)  # [batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  # [batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05)  # [batch_size, n_node]
        node_score = node_scores.unsqueeze(2)  # [batch_size, n_node, 1]

        _batch_size, _n_nodes = node_type_ids.size()

        # Embed type
        T = make_one_hot(node_type_ids.view(-1).contiguous(), self.config.n_ntype).view(
            _batch_size, _n_nodes, self.config.n_ntype
        )
        node_type_emb = self.activation(self.emb_node_type(T))  # [batch_size, n_node, dim/2]

        # Embed score
        if self.basis_f == "sin":
            js = (
                torch.arange(self.hidden_size // 2).unsqueeze(0).unsqueeze(0).float().to(node_type_ids.device)
            )  # [1,1,dim/2]
            js = torch.pow(1.1, js)  # [1,1,dim/2]
            B = torch.sin(js * node_score)  # [batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B))  # [batch_size, n_node, dim/2]
        elif self.basis_f == "id":
            B = node_score
            node_score_emb = self.activation(self.emb_score(B))  # [batch_size, n_node, dim/2]
        elif self.basis_f == "linact":
            B = self.activation(self.B_lin(node_score))  # [batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B))  # [batch_size, n_node, dim/2]

        X = H
        _X = X.view(-1, X.size(2)).contiguous()  # [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _node_type = node_type_ids.view(-1).contiguous()  # [`total_n_nodes`, ]
        _node_feature_extra = (
            torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous()
        )  # [`total_n_nodes`, dim]

        # Merged core
        encoder_outputs, _X = self.encoder(
            embedding_output,
            extended_attention_mask,
            special_tokens_mask,
            head_mask,
            _X,
            edge_index,
            edge_type,
            _node_type,
            _node_feature_extra,
            special_nodes_mask,
            output_hidden_states=output_hidden_states,
        )

        # LM outputs
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here

        # GNN outputs
        X = _X.view(node_type_ids.size(0), node_type_ids.size(1), -1)  # [batch_size, n_node, dim]

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        return outputs, output


@add_start_docstrings(
    """
    GreaseLM Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    GREASELM_START_DOCSTRING,
)
class GreaseLMForMultipleChoice(GreaseLMPreTrainedModel):

    def __init__(self, config, pretrained_concept_emb_file=None):
        super().__init__(config)
        self.greaselm = GreaseLMModel(config, pretrained_concept_emb_file=pretrained_concept_emb_file)
        self.pooler = (
            MultiheadAttPoolLayer(config.n_attention_head, config.hidden_size, config.concept_dim)
            if config.k >= 0
            else None
        )

        concat_vec_dim = config.concept_dim * 2 + config.hidden_size
        self.fc = MLP(concat_vec_dim, config.fc_dim, 1, config.n_fc_layer, config.p_fc, layer_norm=True)

        self.dropout_fc = nn.Dropout(config.p_fc)
        self.layer_id = config.layer_id
        self.init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        if os.path.isdir(pretrained_model_name_or_path):
            concept_emb = os.path.join(pretrained_model_name_or_path, _CONCEPT_EMBEDDINGS_FILE_NAME)
        else:
            concept_emb = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=_CONCEPT_EMBEDDINGS_FILE_NAME, **kwargs
            )
        kwargs["pretrained_concept_emb_file"] = concept_emb
        return super(GreaseLMForMultipleChoice, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        special_tokens_mask,
        concept_ids,
        node_type_ids,
        node_scores,
        adj_lengths,
        special_nodes_mask,
        edge_index,
        edge_type,
        emb_data=None,
        detail=False,
        cache_output=False,
    ):
        """
        :param input_ids:
              (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, seq_len)`):
                   Input ids for the language model.
        :param attention_mask:
              (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, seq_len)`):
                   Attention mask for the language model.
        :param token_type_ids:
              (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, seq_len)`):
                   Token type ids for the language model.
        :param special_tokens_mask:
              (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, seq_len)`):
                   Output mask for the language model.
        :param concept_ids:
              (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, max_node_num)`):
                   Resolved conceptnet ids.
        :param node_type_ids:
              (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, max_node_num)`):
                   Conceptnet id types where 0 == question entity; 1 == answer choice entity; 2 == other node; 3 ==
                   context node
        :param node_scores:
              (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, number_of_choices, max_node_num, 1)`):
                   LM relevancy scores for each resolved conceptnet id.
        :param adj_lengths:
              (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices)`):
                   Adjacency matrix lengths for each batch sample.
        :param special_nodes_mask:
              (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, number_of_choices, max_node_num)`):
                   Mask identifying special nodes in the graph (interaction node in the GreaseLM paper).
        :param edge_index:
               torch.tensor(2, E)) where E is the total number of edges in the particular graph.
        :param edge_type:
               torch.tensor(E, ) where E is the total number of edges in the particular graph.
        :param emb_data:
               torch.tensor(batch_size, number_of_choices, max_node_num, emb_dim)
        :param detail:
               (bool): Whether to return detailed output.
        :param cache_output:
               Whether to cache the output of the language model.
        """
        bs, nc = input_ids.shape[0:2]

        # Merged core
        outputs, gnn_output = self.greaselm(
            input_ids,
            attention_mask,
            token_type_ids,
            special_tokens_mask,
            concept_ids,
            node_type_ids,
            node_scores,
            adj_lengths,
            special_nodes_mask,
            edge_index,
            edge_type,
        )
        # outputs: ([bs, seq_len, sent_dim], [bs, sent_dim], ([bs, seq_len, sent_dim] for _ in range(25)))
        # gnn_output: [bs, n_node, dim_node]

        # LM outputs
        all_hidden_states = outputs[-1]  # ([bs, seq_len, sent_dim] for _ in range(25))
        hidden_states = all_hidden_states[self.layer_id]  # [bs, seq_len, sent_dim]

        sent_vecs = self.greaselm.pooler(hidden_states)  # [bs, sent_dim]

        # GNN outputs
        Z_vecs = gnn_output[:, 0]  # (batch_size, dim_node)

        mask = torch.arange(node_type_ids.size(-1), device=node_type_ids.device) >= adj_lengths[0, ...].unsqueeze(-1)

        mask = mask | (node_type_ids[0, ...] == 3)  # pool over all KG nodes (excluding the context node)
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        graph_vecs, pool_attn = self.pooler(sent_vecs, gnn_output, mask)
        # graph_vecs: [bs, node_dim]

        concat = torch.cat((graph_vecs, sent_vecs, Z_vecs), 1)
        logits = self.fc(self.dropout_fc(concat))

        logits = logits.view(bs, nc)
        return MultipleChoiceModelOutput(
            logits=logits,
            attentions=pool_attn,
        )


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
