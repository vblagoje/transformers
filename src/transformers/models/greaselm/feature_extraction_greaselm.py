# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for GreaseLM."""
import itertools
import json
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

import networkx as nx
import spacy
from spacy.matcher import Matcher
from huggingface_hub import hf_hub_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin, PreTrainedFeatureExtractor
from ...utils import TensorType, logging
from ...utils.logging import get_verbosity, tqdm


blacklist = [
    "-PRON-",
    "actually",
    "likely",
    "possibly",
    "want",
    "make",
    "my",
    "someone",
    "sometimes_people",
    "sometimes",
    "would",
    "want_to",
    "one",
    "something",
    "sometimes",
    "everybody",
    "somebody",
    "could",
    "could_be",
]

merged_relations = [
    "antonym",
    "atlocation",
    "capableof",
    "causes",
    "createdby",
    "isa",
    "desires",
    "hassubevent",
    "partof",
    "hascontext",
    "hasproperty",
    "madeof",
    "notcapableof",
    "notdesires",
    "receivesaction",
    "relatedto",
    "usedfor",
]


"""
English stop words taken from NLTK and expanded to avoid NLTK dependency
For more details see https://github.com/nltk/nltk#copyright , https://github.com/nltk/nltk/blob/develop/LICENSE.txt and 
https://github.com/nltk/nltk/blob/develop/AUTHORS.md
"""
nltk_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                  'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                  'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                  'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                  'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                  'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                  'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                  'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                  'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                  'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                  "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                  'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                  "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

logger = logging.get_logger(__name__)


class GreaseLMFeatureExtractor(FeatureExtractionMixin):
    r"""
    Constructs a GreaseLM feature extractor.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
    """

    model_input_names = [""]

    def __init__(
        self,
        cpnet_vocab_path: Union[Path, str],
        pattern_path: Union[Path, str],
        pruned_graph_path: Union[Path, str],
        score_model: Union[Path, str] = "roberta-large",
        device: str = "cuda",
        cxt_node_connects_all: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cpnet_vocab_path = cpnet_vocab_path
        self.pattern_path = pattern_path
        self.pruned_graph_path = pruned_graph_path
        self.score_model = score_model
        self.device = device

        self.nlp = None
        self.matcher = None
        self.tokenizer = None
        self.model = None
        self.loss_fct = None
        self.cxt_node_connects_all = cxt_node_connects_all

        self.cpnet_vocab = None
        self.cpnet_vocab_underscores = None
        self.concept2id = None
        self.id2relation = None
        self.id2concept = None
        self.relation2id = None

        self.cpnet_simple = None
        self.cpnet = None

    def __call__(
        self,
        question_answer_example: List[Dict[str, Any]],
        entailed_question_answer_example: List[Dict[str, Any]],
        num_choices: int = 5,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main method to prepare for the model one or several image(s).

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
        """
        batch_features = []
        for question_answer_example, entailed_statement in zip(
            question_answer_example, entailed_question_answer_example
        ):
            grouned_statements = self.ground(entailed_statement)
            example_features = self.generate_adj_data_from_grounded_concepts__use_lm(question_answer_example,
                                                                                     grouned_statements)
            batch_features.extend(example_features)
        return self.load_sparse_adj_data_with_contextnode(batch_features, num_choices)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> PreTrainedFeatureExtractor:

        # get preprocessor_config.json
        feature_extractor_dict, kwargs = cls.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)

        # download files needed for preprocessor
        cpnet_vocab_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=feature_extractor_dict["cpnet_vocab_path"], **kwargs
        )
        pattern_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=feature_extractor_dict["pattern_path"], **kwargs
        )
        pruned_graph_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=feature_extractor_dict["pruned_graph_path"], **kwargs
        )

        # set local files as parameters for preprocessor init method
        feature_extractor_dict["cpnet_vocab_path"] = cpnet_vocab_path
        feature_extractor_dict["pattern_path"] = pattern_path
        feature_extractor_dict["pruned_graph_path"] = pruned_graph_path

        return cls.from_dict(feature_extractor_dict, **kwargs)

    def start(self) -> None:
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
        self.nlp.add_pipe("sentencizer")

        self.matcher = Matcher(vocab=self.nlp.vocab)

        self.tokenizer = AutoTokenizer.from_pretrained(self.score_model)
        self.model = AutoModelForMaskedLM.from_pretrained(self.score_model)
        self.model.to(self.device)
        self.model.eval()

        self.loss_fct = CrossEntropyLoss(reduction="none")

        with open(self.cpnet_vocab_path, "r", encoding="utf8") as f:
            file_contents = [line.strip() for line in f]
        self.cpnet_vocab = [c.replace("_", " ") for c in file_contents]
        self.cpnet_vocab_underscores = [l for l in file_contents]
        with open(self.pattern_path, "r", encoding="utf8") as fin:
            all_patterns = json.load(fin)

        pb = tqdm(total=len(all_patterns), desc="Starting GreaseLMFeatureExtractor")
        for concept, pattern in all_patterns.items():
            self.matcher.add(concept, [pattern])
            pb.update(1)

        with open(self.cpnet_vocab_path, "r", encoding="utf8") as fin:
            self.id2concept = [w.strip() for w in fin]
        self.concept2id = {w: i for i, w in enumerate(self.id2concept)}

        self.id2relation = merged_relations
        self.relation2id = {r: i for i, r in enumerate(self.id2relation)}
        self.cpnet = nx.read_gpickle(self.pruned_graph_path)
        self.cpnet_simple = nx.Graph()
        for u, v, data in self.cpnet.edges(data=True):
            w = data["weight"] if "weight" in data else 1.0
            if self.cpnet_simple.has_edge(u, v):
                self.cpnet_simple[u][v]["weight"] += w
            else:
                self.cpnet_simple.add_edge(u, v, weight=w)
        logger.info("GreaseLMFeatureExtractor started")

    def lemmatize(self, concept: str):

        doc = self.nlp(concept.replace("_", " "))
        lcs = set()
        lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
        return lcs

    def ground_mentioned_concepts(self, sentence: str, answer: str = None):

        sentence = sentence.lower()
        doc = self.nlp(sentence)
        matches = self.matcher(doc)

        mentioned_concepts = set()
        span_to_concepts = {}

        if answer is not None:
            ans_matcher = Matcher(self.nlp.vocab)
            ans_words = self.nlp(answer)
            # print(ans_words)
            ans_matcher.add(answer, [[{"TEXT": token.text.lower()} for token in ans_words]])

            ans_match = ans_matcher(doc)
            ans_mentions = set()
            for _, ans_start, ans_end in ans_match:
                ans_mentions.add((ans_start, ans_end))

        for match_id, start, end in matches:
            if answer is not None:
                if (start, end) in ans_mentions:
                    continue

            span = doc[start:end].text  # the matched span

            # a word that appears in answer is not considered as a mention in the question
            # if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
            #     continue
            original_concept = self.nlp.vocab.strings[match_id]
            original_concept_set = set()
            original_concept_set.add(original_concept)

            # print("span", span)
            # print("concept", original_concept)
            # print("Matched '" + span + "' to the rule '" + string_id)

            # why do you lemmatize a mention whose len == 1?

            if len(original_concept.split("_")) == 1:
                # tag = doc[start].tag_
                # if tag in ['VBN', 'VBG']:

                original_concept_set.update(self.lemmatize(self.nlp.vocab.strings[match_id]))

            if span not in span_to_concepts:
                span_to_concepts[span] = set()

            span_to_concepts[span].update(original_concept_set)

        for span, concepts in span_to_concepts.items():
            concepts_sorted = list(concepts)
            # print("span:")
            # print(span)
            # print("concept_sorted:")
            # print(concepts_sorted)
            concepts_sorted.sort(key=len)

            # mentioned_concepts.update(concepts_sorted[0:2])

            shortest = concepts_sorted[0:3]

            for c in shortest:
                if c in blacklist:
                    continue

                # a set with one string like: set("like_apples")
                lcs = self.lemmatize(c)
                intersect = lcs.intersection(shortest)
                if len(intersect) > 0:
                    mentioned_concepts.add(list(intersect)[0])
                else:
                    mentioned_concepts.add(c)

            # if a mention exactly matches with a concept

            exact_match = set(
                [concept for concept in concepts_sorted if concept.replace("_", " ").lower() == span.lower()]
            )
            # print("exact match:")
            # print(exact_match)
            assert len(exact_match) < 2
            mentioned_concepts.update(exact_match)
        return mentioned_concepts

    def ground_qa_pair(self, statement: str, answer: str):
        all_concepts = self.ground_mentioned_concepts(statement, answer)
        answer_concepts = self.ground_mentioned_concepts(answer)
        question_concepts = all_concepts - answer_concepts
        if len(question_concepts) == 0:
            question_concepts = self.hard_ground(statement)  # not very possible

        if len(answer_concepts) == 0:
            answer_concepts = self.hard_ground(answer)  # some case

        # question_concepts = question_concepts -  answer_concepts
        question_concepts = sorted(list(question_concepts))
        answer_concepts = sorted(list(answer_concepts))
        return {"sent": statement, "ans": answer, "qc": question_concepts, "ac": answer_concepts}

    def hard_ground(self, sent: str):
        sent = sent.lower()
        doc = self.nlp(sent)
        res = set()
        for t in doc:
            if t.lemma_ in self.cpnet_vocab:
                res.add(t.lemma_)
        sent = " ".join([t.text for t in doc])
        if sent in self.cpnet_vocab:
            res.add(sent)
        try:
            assert len(res) > 0
        except Exception:
            print(f"for {sent}, concept not found in hard grounding.")
        return res

    def match_mentioned_concepts(self, statements: List[str], answers: List[str]):
        grounded_examples: List[Dict[str]] = []
        for statement, answer in zip(statements, answers):
            grounded_examples.append(self.ground_qa_pair(statement, answer))
        return grounded_examples

    # To-do: examine prune
    def prune(self, grounded_examples: List[Dict[str, str]]):
        prune_examples = []
        for item in grounded_examples:
            qc = item["qc"]
            prune_qc = []
            for c in qc:
                if c[-2:] == "er" and c[:-2] in qc:
                    continue
                if c[-1:] == "e" and c[:-1] in qc:
                    continue
                have_stop = False
                # remove all concepts having stopwords, including hard-grounded ones
                for t in c.split("_"):
                    if t in nltk_stopwords:
                        have_stop = True
                if not have_stop and c in self.cpnet_vocab_underscores:
                    prune_qc.append(c)

            ac = item["ac"]
            prune_ac = []
            for c in ac:
                if c[-2:] == "er" and c[:-2] in ac:
                    continue
                if c[-1:] == "e" and c[:-1] in ac:
                    continue
                all_stop = True
                for t in c.split("_"):
                    if t not in nltk_stopwords:
                        all_stop = False
                if not all_stop and c in self.cpnet_vocab_underscores:
                    prune_ac.append(c)

            try:
                assert len(prune_ac) > 0 and len(prune_qc) > 0
            except Exception as e:
                pass
                # print("In pruning")
                # print(prune_qc)
                # print(prune_ac)
                # print("original:")
                # print(qc)
                # print(ac)
                # print()
            item["qc"] = prune_qc
            item["ac"] = prune_ac

            prune_examples.append(item)
        return prune_examples

    def ground(self, common_sense_example: Dict) -> Dict:

        # common_sense_example: Dict
        """
        {'answerKey': 'B',
         'id': 'b8c0a4703079cf661d7261a60a1bcbff', 'question': {'question_concept': 'magazines',
                      'choices': [{'label': 'A', 'text': 'doctor'}, {'label': 'B', 'text': 'bookstore'},
                                  {'label': 'C', 'text': 'market'}, {'label': 'D', 'text': 'train station'}, {'label':
                                  'E', 'text': 'mortuary'}],
                      'stem': 'Where would you find magazines along side many other printed works?'},
         'statements': [
             {'label': False, 'statement': 'Doctor would you find magazines along side many other printed works.'},
             {'label': True, 'statement': 'Bookstore would you find magazines along side many other printed works.'},
             {'label': False, 'statement': 'Market would you find magazines along side many other printed works.'},
             {'label': False,
              'statement': 'Train station would you find magazines along side many other printed works.'},
             {'label': False, 'statement': 'Mortuary would you find magazines along side many other printed works.'}]}
        """
        statements = []
        answers = []

        for statement in common_sense_example["statements"]:
            statements.append(statement["statement"])

        for answer in common_sense_example["question"]["choices"]:
            ans = answer["text"]
            try:
                assert all([i != "_" for i in ans])
            except Exception:
                print(ans)
            answers.append(ans)

        grounded_concepts = self.match_mentioned_concepts(statements, answers)
        res = self.prune(grounded_concepts)

        return res

    def concepts2adj(self, node_ids):
        cids = np.array(node_ids, dtype=np.int32)
        n_rel = len(self.id2relation)
        n_node = cids.shape[0]
        adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
        for s in range(n_node):
            for t in range(n_node):
                s_c, t_c = cids[s], cids[t]
                if self.cpnet.has_edge(s_c, t_c):
                    for e_attr in self.cpnet[s_c][t_c].values():
                        if 0 <= e_attr["rel"] < n_rel:
                            adj[e_attr["rel"]][s][t] = 1
        # cids += 1  # note!!! index 0 is reserved for padding
        adj = adj.reshape(-1, n_node)
        return adj, cids

    def get_lm_score(self, cids, question):
        cids = cids[:]
        cids.insert(0, -1)  # QAcontext node
        sents, scores = [], []
        for cid in cids:
            if cid == -1:
                sent = question.lower()
            else:
                sent = "{} {}.".format(question.lower(), " ".join(self.id2concept[cid].split("_")))
            sent = self.tokenizer.encode(sent, add_special_tokens=True)
            sents.append(sent)
        n_cids = len(cids)
        cur_idx = 0
        batch_size = 50
        while cur_idx < n_cids:
            # Prepare batch
            input_ids = sents[cur_idx : cur_idx + batch_size]
            max_len = max([len(seq) for seq in input_ids])
            for j, seq in enumerate(input_ids):
                seq += [self.tokenizer.pad_token_id] * (max_len - len(seq))
                input_ids[j] = seq
            input_ids = torch.tensor(input_ids).to(self.device)  # [B, seqlen]
            mask = (input_ids != 1).long()  # [B, seq_len]
            # Get LM score
            with torch.no_grad():
                _scores = self.model_score(input_ids, mask)
            scores += _scores
            cur_idx += batch_size
        assert len(sents) == len(scores) == len(cids)
        cid2score = OrderedDict(sorted(list(zip(cids, scores)), key=lambda x: -x[1]))  # score: from high to low
        return cid2score

    def model_score(self, input_ids, mask):
        output = self.model(input_ids, attention_mask=mask, labels=input_ids)
        bsize, seqlen = input_ids.size()
        prediction = output["logits"].view(-1, self.model.config.vocab_size)
        ground_truth = input_ids.view(-1)
        lm_loss = self.loss_fct(prediction, ground_truth).view(bsize, seqlen)
        scores = (lm_loss * mask).sum(dim=1)
        return list(-scores.detach().cpu().numpy())

    def concepts_to_adj_matrices_2hop_all_pair__use_lm__part1(self, data):
        qc_ids, ac_ids, question = data
        qa_nodes = set(qc_ids) | set(ac_ids)
        extra_nodes = set()
        for qid in qa_nodes:
            for aid in qa_nodes:
                if qid != aid and qid in self.cpnet_simple.nodes and aid in self.cpnet_simple.nodes:
                    extra_nodes |= set(self.cpnet_simple[qid]) & set(self.cpnet_simple[aid])
        extra_nodes = extra_nodes - qa_nodes
        return sorted(qc_ids), sorted(ac_ids), question, sorted(extra_nodes)

    def concepts_to_adj_matrices_2hop_all_pair__use_lm__part2(self, data):
        qc_ids, ac_ids, question, extra_nodes = data
        cid2score = self.get_lm_score(qc_ids + ac_ids + extra_nodes, question)
        return qc_ids, ac_ids, question, extra_nodes, cid2score

    def concepts_to_adj_matrices_2hop_all_pair__use_lm__part3(self, data):
        qc_ids, ac_ids, question, extra_nodes, cid2score = data
        schema_graph = qc_ids + ac_ids + sorted(extra_nodes, key=lambda x: -cid2score[x])  # score: from high to low
        arange = np.arange(len(schema_graph))
        qmask = arange < len(qc_ids)
        amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
        adj, concepts = self.concepts2adj(schema_graph)
        return {"adj": adj, "concepts": concepts, "qmask": qmask, "amask": amask, "cid2score": cid2score}

    def generate_adj_data_from_grounded_concepts__use_lm(self, statement, grounded_statements) -> List[Dict[str, Any]]:
        """
        This function will save
            (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix) (2) concepts ids (3) qmask that
            specifices whether a node is a question concept (4) amask that specifices whether a node is a answer
            concept (5) cid2score that maps a concept id to its relevance score given the QA context
        to the output path in python pickle format

        grounded_path: str cpnet_graph_path: str cpnet_vocab_path: str output_path: str
        """

        qa_data = []
        for grounded_statement in grounded_statements:
            q_ids = set(self.concept2id[c] for c in grounded_statement["qc"])
            a_ids = set(self.concept2id[c] for c in grounded_statement["ac"])
            q_ids = q_ids - a_ids
            qa_context = "{} {}.".format(statement["question"]["stem"], grounded_statement["ans"])
            qa_data.append((q_ids, a_ids, qa_context))

        choice_features = []
        for qa_item in qa_data:
            qa_item_adj = self.concepts_to_adj_matrices_2hop_all_pair__use_lm__part1(qa_item)
            qa_item_adj_scored = self.concepts_to_adj_matrices_2hop_all_pair__use_lm__part2(qa_item_adj)
            graph_features = self.concepts_to_adj_matrices_2hop_all_pair__use_lm__part3(qa_item_adj_scored)
            choice_features.append(graph_features)
        return choice_features

    def load_sparse_adj_data_with_contextnode(
        self,
        adj_concept_pairs,
        num_choices,
        concepts_by_sents_list=None,
        disable_tqdm=bool(get_verbosity() >= logging.WARNING),
    ) -> Dict[str, Any]:
        """Construct input tensors for the GNN component of the model."""
        # Set special nodes and links
        context_node = 0
        n_special_nodes = 1
        cxt2qlinked_rel = 0
        cxt2alinked_rel = 1
        half_n_rel = len(self.id2relation) + 2
        if self.cxt_node_connects_all:
            cxt2other_rel = half_n_rel
            half_n_rel += 1

        n_samples = len(adj_concept_pairs)  # this is actually n_questions x n_choices
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
        concept_ids = torch.full((n_samples, self.max_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, self.max_node_num), 2, dtype=torch.long)  # default 2: "other node"
        node_scores = torch.zeros((n_samples, self.max_node_num, 1), dtype=torch.float)
        special_nodes_mask = torch.zeros(n_samples, self.max_node_num, dtype=torch.bool)

        adj_lengths_ori = adj_lengths.clone()
        if not concepts_by_sents_list:
            concepts_by_sents_list = itertools.repeat(None)
        for idx, (_data, cpts_by_sents) in tqdm(
            enumerate(zip(adj_concept_pairs, concepts_by_sents_list)),
            total=n_samples,
            desc="loading adj matrices",
            disable=disable_tqdm,
        ):
            adj, concepts, qm, am, cid2score = (
                _data["adj"],
                _data["concepts"],
                _data["qmask"],
                _data["amask"],
                _data["cid2score"],
            )

            assert n_special_nodes <= self.max_node_num
            special_nodes_mask[idx, :n_special_nodes] = 1
            num_concept = min(
                len(concepts) + n_special_nodes, self.max_node_num
            )  # this is the final number of nodes including contextnode but excluding PAD
            adj_lengths_ori[idx] = len(concepts)
            adj_lengths[idx] = num_concept

            # Prepare nodes
            concepts = concepts[: num_concept - n_special_nodes]
            concept_ids[idx, n_special_nodes:num_concept] = torch.tensor(
                concepts + 1
            )  # To accomodate contextnode, original concept_ids incremented by 1
            concept_ids[idx, 0] = context_node  # this is the "concept_id" for contextnode

            # Prepare node scores
            if cid2score is not None:
                if -1 not in cid2score:
                    cid2score[-1] = 0
                for _j_ in range(num_concept):
                    _cid = int(concept_ids[idx, _j_]) - 1  # Now context node is -1
                    node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

            # Prepare node types
            node_type_ids[idx, 0] = 3  # context node
            node_type_ids[idx, 1:n_special_nodes] = 4  # sent nodes
            node_type_ids[idx, n_special_nodes:num_concept][
                torch.tensor(qm, dtype=torch.bool)[: num_concept - n_special_nodes]
            ] = 0
            node_type_ids[idx, n_special_nodes:num_concept][
                torch.tensor(am, dtype=torch.bool)[: num_concept - n_special_nodes]
            ] = 1

            # Load adj
            elements = np.nonzero(adj)  # elements are (row, col)
            ij = torch.tensor(elements[0], dtype=torch.int64)  # (num_matrix_entries, ), where each entry is coordinate
            k = torch.tensor(elements[1], dtype=torch.int64)  # (num_matrix_entries, ), where each entry is coordinate
            n_node = adj.shape[1]
            assert len(self.id2relation) == adj.shape[0] // n_node
            i, j = torch.div(ij, n_node, rounding_mode="floor"), ij % n_node

            # Prepare edges
            i += 2
            j += 1
            k += 1  # **** increment coordinate by 1, rel_id by 2 ****
            extra_i, extra_j, extra_k = [], [], []
            for _coord, q_tf in enumerate(qm):
                _new_coord = _coord + n_special_nodes
                if _new_coord > num_concept:
                    break
                if q_tf:
                    extra_i.append(cxt2qlinked_rel)  # rel from contextnode to question concept
                    extra_j.append(0)  # contextnode coordinate
                    extra_k.append(_new_coord)  # question concept coordinate
                elif self.cxt_node_connects_all:
                    extra_i.append(cxt2other_rel)  # rel from contextnode to other concept
                    extra_j.append(0)  # contextnode coordinate
                    extra_k.append(_new_coord)  # other concept coordinate
            for _coord, a_tf in enumerate(am):
                _new_coord = _coord + n_special_nodes
                if _new_coord > num_concept:
                    break
                if a_tf:
                    extra_i.append(cxt2alinked_rel)  # rel from contextnode to answer concept
                    extra_j.append(0)  # contextnode coordinate
                    extra_k.append(_new_coord)  # answer concept coordinate
                elif self.cxt_node_connects_all:
                    extra_i.append(cxt2other_rel)  # rel from contextnode to other concept
                    extra_j.append(0)  # contextnode coordinate
                    extra_k.append(_new_coord)  # other concept coordinate

            # half_n_rel += 2 #should be 19 now
            if len(extra_i) > 0:
                i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                k = torch.cat([k, torch.tensor(extra_k)], dim=0)
            ########################

            mask = (j < self.max_node_num) & (k < self.max_node_num)
            i, j, k = i[mask], j[mask], k[mask]
            i, j, k = (
                torch.cat((i, i + half_n_rel), 0),
                torch.cat((j, k), 0),
                torch.cat((k, j), 0),
            )  # add inverse relations
            edge_index.append(torch.stack([j, k], dim=0))  # each entry is [2, E]
            edge_type.append(i)  # each entry is [E, ]

        # list of size (n_questions, n_choices), where each entry is tensor[2, E]
        edge_index = list(map(list, zip(*(iter(edge_index),) * num_choices)))
        # list of size (n_questions, n_choices), where each entry is tensor[E, ]
        edge_type = list(map(list, zip(*(iter(edge_type),) * num_choices)))

        concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask = [
            x.view(-1, num_choices, *x.size()[1:])
            for x in (concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask)
        ]

        # concept_ids: (n_questions, num_choice, self.max_node_num)
        # node_type_ids: (n_questions, num_choice, self.max_node_num)
        # node_scores: (n_questions, num_choice, self.max_node_num, 1)
        # adj_lengths: (n_questions,　num_choice)
        # special_nodes_mask: (n_questions, num_choice, self.max_node_num)

        # edge_index: list of size (n_questions, n_choices), where each entry is tensor[2, E]
        # edge_type: list of size (n_questions, n_choices), where each entry is tensor[E, ]
        # We can't stack edge_index and edge_type lists of tensors as tensors are not of equal size
        return dict(
            concept_ids=concept_ids,
            node_type_ids=node_type_ids,
            node_scores=node_scores,
            adj_lengths=adj_lengths,
            special_nodes_mask=special_nodes_mask,
            edge_index=edge_index,
            edge_type=edge_type,
        )
