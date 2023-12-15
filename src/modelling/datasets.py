#          Linking Surface Facts to Large-Scale Knowledge Graphs
# 
#   file: datasets.py
# 
# Authors: Gorjan Radevski (gorjanradevski@gmail.com) 
#          Kiril Gashteovski (kiril.gashteovski@neclab.eu)
#    	   Chia-Chien Hung (Chia-Chien.Hung@neclab.eu) 
#          Carolin Lawrence (carolin.lawrence@neclab.eu)
#          Goran Glavas (goran.glavas@uni-wuerzburg.de)
#
# NEC Laboratories Europe GmbH, Copyright (c) 2023, All rights reserved. 
#     THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
# 
#          PROPRIETARY INFORMATION --- 
#
# SOFTWARE LICENSE AGREEMENT
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
# 
# This is a license agreement ("Agreement") between your academic institution or non-profit organization or self (called "Licensee" or "You" in this Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this Agreement).  All rights not specifically granted to you in this Agreement are reserved for Licensor. 
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive ownership of any copy of the Software (as defined below) licensed under this Agreement and hereby grants to Licensee a personal, non-exclusive, non-transferable license to use the Software for noncommercial research purposes, without the right to sublicense, pursuant to the terms and conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF LICENSOR’S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this Agreement, the term "Software" means (i) the actual copy of all or any portion of code for program routines made accessible to Licensee by Licensor pursuant to this Agreement, inclusive of backups, updates, and/or merged copies permitted hereunder or subsequently supplied by Licensor,  including all or any file structures, programming instructions, user interfaces and screen formats and sequences as well as any and all documentation and instructions related to it, and (ii) all or any derivatives and/or modifications created or made by You to any of the items specified in (i).
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is proprietary to Licensor, and as such, Licensee agrees to receive all such materials and to use the Software only in accordance with the terms of this Agreement.  Licensee agrees to use reasonable effort to protect the Software from unauthorized use, reproduction, distribution, or publication. All publication materials mentioning features or use of this software must explicitly include an acknowledgement the software was developed by NEC Laboratories Europe GmbH.
# COPYRIGHT: The Software is owned by Licensor.  
# PERMITTED USES:  The Software may be used for your own noncommercial internal research purposes. You understand and agree that Licensor is not obligated to implement any suggestions and/or feedback you might provide regarding the Software, but to the extent Licensor does so, you are not entitled to any compensation related thereto.
# DERIVATIVES: You may create derivatives of or make modifications to the Software, however, You agree that all and any such derivatives and modifications will be owned by Licensor and become a part of the Software licensed to You under this Agreement.  You may only use such derivatives and modifications for your own noncommercial internal research purposes, and you may not otherwise use, distribute or copy such derivatives and modifications in violation of this Agreement.
# BACKUPS:  If Licensee is an organization, it may make that number of copies of the Software necessary for internal noncommercial use at a single site within its organization provided that all information appearing in or on the original labels, including the copyright and trademark notices are copied onto the labels of the copies.
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except as explicitly permitted herein. Licensee has not been granted any trademark license as part of this Agreement. Neither the name of NEC Laboratories Europe GmbH nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in whole or in part, or provide third parties access to prior or present versions (or any parts thereof) of the Software.
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder without the prior written consent of Licensor. Any attempted assignment without such consent shall be null and void.
# TERM: The term of the license granted by this Agreement is from Licensee's acceptance of this Agreement by downloading the Software or by using the Software until terminated as provided below.
# The Agreement automatically terminates without notice if you fail to comply with any provision of this Agreement.  Licensee may terminate this Agreement by ceasing using the Software.  Upon any termination of this Agreement, Licensee will delete any and all copies of the Software. You agree that all provisions which operate to protect the proprietary rights of Licensor shall remain in force should breach occur and that the obligation of confidentiality described in this Agreement is binding in perpetuity and, as such, survives the term of the Agreement.
# FEE: Provided Licensee abides completely by the terms and conditions of this Agreement, there is no fee due to Licensor for Licensee's use of the Software in accordance with this Agreement.
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON-INFRINGEMENT.  LICENSEE BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND RELATED MATERIALS.
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is provided as part of this Agreement.  
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent permitted under applicable law, Licensor shall not be liable for direct, indirect, special, incidental, or consequential damages or lost profits related to Licensee's use of and/or inability to use the Software, even if Licensor is advised of the possibility of such damage.
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable export control laws, regulations, and/or other laws related to embargoes and sanction programs administered by law.
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be invalid, illegal, or unenforceable by a court or other tribunal of competent jurisdiction, the validity, legality and enforceability of the remaining provisions shall not in any way be affected or impaired thereby.
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right or remedy under this Agreement shall be construed as a waiver of any future or other exercise of such right or remedy by Licensor.
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance with the laws of Germany without reference to conflict of laws principles.  You consent to the personal jurisdiction of the courts of this country and waive their rights to venue outside of Germany.
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and entire agreement between Licensee and Licensor as to the matter set forth herein and supersedes any previous agreements, understandings, and arrangements between the parties relating hereto.
#      THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

import json
import random
import re
from collections import defaultdict
from copy import deepcopy
from os.path import join as pjoin

import numpy as np
import torch
from transformers import AutoTokenizer, RobertaTokenizerFast
from yacs.config import CfgNode

from utils.wikidata import Wikidata


class BaseDataset:
    def __init__(self, cfg: CfgNode, mode: str):
        self.cfg = cfg
        assert mode in ["train", "val", "test"]
        dataset_path = (
            self.cfg.TRAIN_DATASET_PATH
            if mode == "train"
            else self.cfg.VAL_DATASET_PATH
            if mode == "val"
            else self.cfg.TEST_DATASET_PATH
        )
        self.dataset = json.load(open(dataset_path))

    def __len__(self):
        return len(self.dataset)

    def replacer(self, sentence: str, to_be_replaced: str, replacer: str):
        # If using full sentences, this function is being used
        return re.sub(
            rf"(?!\B\w)({re.escape(to_be_replaced) + '(s|es)?'})(?<!\w\B)",
            replacer,
            sentence,
            count=1,
        )

    def __getitem__(self, idx: int):
        # Prepare sentence
        sentence = self.dataset[idx]["sentence"]
        # Prepare extraction
        e_subj, e_rel, e_obj = self.dataset[idx]["extraction"]
        oie = f"<SUBJ> {e_subj} <PRED> {e_rel} <OBJ> {e_obj}"
        # Prepare fact
        f_subj, f_rel, f_obj = self.dataset[idx]["fact_ids"]
        # Check for sentence context
        if self.cfg.SENTENCE_CONTEXT:
            # Obtain original extraction: it will be found in the context sentence
            org_subj, _, org_obj = self.dataset[idx]["org_extraction"]
            # Replace subject in context sentence
            try:
                sentence = self.replacer(
                    sentence, to_be_replaced=org_subj, replacer=e_subj
                )
            except re.error:
                pass
            try:
                # Replace object in context sentence
                sentence = self.replacer(
                    sentence, to_be_replaced=org_obj, replacer=e_obj
                )
            except re.error:
                pass
            oie = f"{oie} <SENT> {sentence}"

        return {
            "sentence": sentence,
            "oie": oie,
            "fact_ids": {"subj": f_subj, "rel": f_rel, "obj": f_obj},
            "org_extraction": self.dataset[idx]["org_extraction"],
        }


class BaseCollater:
    # Also works as an inference collater
    def __init__(self, cfg: CfgNode):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        special_tokens = ["<SUBJ>", "<PRED>", "<OBJ>", "<ENT>", "<REL>"]
        if cfg.SENTENCE_CONTEXT:
            special_tokens.append("<SENT>")
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def __call__(self, batch):
        # Prepare output dictionary
        output = defaultdict(list)
        # Transfer what's needed to the batch
        for b in batch:
            for k in b.keys():
                output[k].append(b[k])
        # Prepare sentences
        output["sentence_input"] = self.tokenizer(
            output["sentence"],
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=32,
        )
        # Prepare extractions
        output["oie_input"] = self.tokenizer(
            output["oie"],
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        # Get extraction pooling indices
        batch_size = output["oie_input"]["input_ids"].size(0)
        output["oie_pooling_indices"] = torch.zeros(batch_size, 3, dtype=torch.int)
        # First 3 tokens are relevant for the extraction - somewhat of a hacky solution
        for i, token_id in enumerate(self.tokenizer.additional_special_tokens_ids[:3]):
            indices = torch.where(output["oie_input"]["input_ids"] == token_id)[1]
            output["oie_pooling_indices"][:, i] = indices

        return output


class OokgDetectionCollater(BaseCollater):
    def __init__(self, cfg: CfgNode, wikidata: Wikidata):
        super(OokgDetectionCollater, self).__init__()
        self.cfg = cfg
        self.wikidata = wikidata
        # For graph negative sampling
        self.entity_embeddings = np.load(
            pjoin(self.cfg.INDEX_PATH, "entity_embeddings.npy")
        )
        self.entity2index = {
            v: k
            for k, v in json.load(
                open(pjoin(self.cfg.INDEX_PATH, "index2entity.json"))
            ).items()
        }
        self.rel_embeddings = np.load(
            pjoin(self.cfg.INDEX_PATH, "relation_embeddings.npy")
        )
        self.rel2index = {
            v: k
            for k, v in json.load(
                open(pjoin(self.cfg.INDEX_PATH, "index2rel.json"))
            ).items()
        }
        self.all_entities = list(self.entity2index.keys())
        self.all_relations = list(self.rel2index.keys())

    def __call__(self, batch):
        # Invoke call to base
        output = BaseCollater.__call__(self, batch)
        # Prepare graph of entities & relations
        unique_entities, unique_relations = set(), set()
        for fact in output["fact_ids"]:
            # Add unique entities
            unique_entities.add(fact["subj"])
            unique_entities.add(fact["obj"])
            # Add unique relations
            unique_relations.add(fact["rel"])
        # (1) Add only negative entities & relations
        entity_embeddings, relation_embeddings = [], []
        sampled = 0
        while sampled < self.cfg.GRAPH_NEGATIVE_ENTITIES:
            entity_id = random.choice(self.all_entities)
            if entity_id in unique_entities:
                continue
            # Get entity data
            index = int(self.entity2index[entity_id])
            entity_embeddings.append(torch.from_numpy(self.entity_embeddings[index]))
            sampled += 1
        sampled = 0
        while sampled < self.cfg.GRAPH_NEGATIVE_RELATIONS:
            rel_id = random.choice(self.all_relations)
            if rel_id in unique_relations:
                continue
            # Get relation data
            index = int(self.rel2index[rel_id])
            relation_embeddings.append(torch.from_numpy(self.rel_embeddings[index]))
            sampled += 1
        # (2) Remove ~50% of the entities & relations
        for e in list(unique_entities):
            if torch.bernoulli(torch.tensor([0.5])).item():
                continue
            unique_entities.remove(e)
        for r in list(unique_relations):
            if torch.bernoulli(torch.tensor([0.5])).item():
                continue
            unique_relations.remove(r)
        output["unique_entities"] = unique_entities
        output["unique_relations"] = unique_relations
        # (3) Add remaining entities & relations
        for e in list(unique_entities):
            index = int(self.entity2index[e])
            entity_embeddings.append(torch.from_numpy(self.entity_embeddings[index]))
        for r in list(unique_relations):
            index = int(self.rel2index[r])
            relation_embeddings.append(torch.from_numpy(self.rel_embeddings[index]))
        output["entity_embeddings"] = torch.stack(entity_embeddings, dim=0)
        output["relation_embeddings"] = torch.stack(relation_embeddings, dim=0)

        return output


class SlotLinkingTrainingCollater(BaseCollater):
    def __init__(self, cfg: CfgNode, wikidata: Wikidata):
        super(SlotLinkingTrainingCollater, self).__init__(cfg=cfg)
        self.cfg = cfg
        self.wikidata = wikidata
        # For graph negative sampling
        self.all_entities = list([e for e in self.wikidata.wikidata if e[0] == "Q"])
        self.all_relations = list([r for r in self.wikidata.wikidata if r[0] == "P"])

    def __call__(self, batch):
        # Invoke call to base
        output = BaseCollater.__call__(self, batch)
        # Prepare graph of entities & relations
        unique_entities, unique_relations = set(), set()
        for fact in output["fact_ids"]:
            # Add unique entities
            unique_entities.add(fact["subj"])
            unique_entities.add(fact["obj"])
            # Add unique relations
            unique_relations.add(fact["rel"])
        # Prepare (negative) entities & relations
        entities_text, relations_text = [], []
        output["entity2index"], output["rel2index"] = {}, {}
        for i, entity_id in enumerate(list(unique_entities)):
            output["entity2index"][entity_id] = i
            # Get entity data
            entity_data = self.wikidata.get_data(entity_id)
            entity_label = self.wikidata.get_label(entity_data)
            entity_desc = self.wikidata.get_description(entity_data)
            entities_text.append(f"{entity_label} <ENT> {entity_desc}")
        for i, rel_id in enumerate(list(unique_relations)):
            output["rel2index"][rel_id] = i
            # Get relation data
            rel_data = self.wikidata.get_data(rel_id)
            rel_label = self.wikidata.get_label(rel_data)
            rel_desc = self.wikidata.get_description(rel_data)
            relations_text.append(f"{rel_label} <REL> {rel_desc}")
        # Add graph negative entities
        if self.cfg.GRAPH_NEGATIVE_ENTITIES > 0:
            sampled = 0
            while sampled < self.cfg.GRAPH_NEGATIVE_ENTITIES:
                entity_id = random.choice(self.all_entities)
                if entity_id in unique_entities:
                    continue
                # Get entity data
                entity_data = self.wikidata.get_data(entity_id)
                entity_label = self.wikidata.get_label(entity_data)
                entity_desc = self.wikidata.get_description(entity_data)
                entities_text.append(f"{entity_label} <ENT> {entity_desc}")
                sampled += 1
        # Add graph negative relations
        if self.cfg.GRAPH_NEGATIVE_RELATIONS > 0:
            sampled = 0
            while sampled < self.cfg.GRAPH_NEGATIVE_RELATIONS:
                rel_id = random.choice(self.all_relations)
                if rel_id in unique_relations:
                    continue
                # Get relation data
                rel_data = self.wikidata.get_data(rel_id)
                rel_label = self.wikidata.get_label(rel_data)
                rel_desc = self.wikidata.get_description(rel_data)
                relations_text.append(f"{rel_label} <REL> {rel_desc}")
                sampled += 1
        # Tokenize entities & relations text
        output["entity_text_input"] = self.tokenizer(
            entities_text,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=32,
        )
        output["rel_text_input"] = self.tokenizer(
            relations_text,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=32,
        )

        return output


class FactRerankDataset:
    def __init__(self, cfg: CfgNode, mode: str, wikidata: Wikidata):
        self.cfg = cfg
        assert mode in ["train", "val", "test"]
        dataset_path = (
            self.cfg.TRAIN_DATASET_PATH
            if mode == "train"
            else self.cfg.VAL_DATASET_PATH
            if mode == "val"
            else self.cfg.TEST_DATASET_PATH
        )
        self.dataset = json.load(open(dataset_path))
        self.wikidata = wikidata
        # Prepare arguments
        self.arguments = {"subj_obj": set(), "rel": set()}
        for k in self.wikidata.wikidata:
            if k[0] == "Q":
                self.arguments["subj_obj"].add(k)
            elif k[0] == "P":
                self.arguments["rel"].add(k)
            else:
                raise ValueError(f"Impossible: {k}")
        self.arguments = {k: list(v) for k, v in self.arguments.items()}
        if self.cfg.SIMILARITIES_PATH:
            self.similarities = json.load(open(self.cfg.SIMILARITIES_PATH))

    def __len__(self):
        return len(self.dataset)

    def get_similar_object(self, obj: str, obj_index: int):
        # Sample bad, within dataset negative
        obj_index2argument = {0: "subj_obj", 1: "rel", 2: "subj_obj"}
        # Perform sampling
        if not hasattr(self, "similarities"):
            return random.choice(self.arguments[obj_index2argument[obj_index]])
        if obj not in self.similarities or len(self.similarities[obj]) == 0:
            return random.choice(self.arguments[obj_index2argument[obj_index]])
        # Else sample hard negative
        objects = [e["obj"] for e in self.similarities[obj]]
        # In order to normalize
        probs = np.array([e["score"] for e in self.similarities[obj]])
        random_obj = np.random.choice(objects, p=probs / probs.sum(), size=1)[0]

        return random_obj

    def __getitem__(self, idx: int):
        # Prepare extraction
        e_subj, e_rel, e_obj = self.dataset[idx]["extraction"]
        # Prepare fact
        fact_ids = deepcopy(self.dataset[idx]["fact_ids"])
        # subj, rel, obj change
        changes = torch.bernoulli(torch.tensor([0.2, 0.2, 0.2]))
        # Perform replacements of subj, rel, obj
        for i, c in enumerate(changes):
            if c:
                fact_ids[i] = self.get_similar_object(fact_ids[i], i)
        # If none are replaced, ground truth is 1.0, else 0.0
        # 0.8 * 0.8 * 0.8 = 0.512 (prob of being negative)
        ground_truth = torch.tensor([1.0]) if not changes.sum() else torch.tensor([0.0])
        # Add descriptions, and convert the fact to text
        fact_text = []
        for i in range(len(fact_ids)):
            arg_id = fact_ids[i]
            data = self.wikidata.get_data(arg_id)
            label = self.wikidata.get_label(data)
            desc = self.wikidata.get_description(data)
            # Masking to put more focus on the descriptions
            if torch.bernoulli(torch.tensor([0.8])):
                label = "<mask>"
            fact_text.append(f"{label} <DESC> {desc}")
        # Convert the oie to text
        oie = f"<SUBJ> {e_subj} <PRED> {e_rel} <OBJ> {e_obj}"
        # Create the fact in a full text form
        fact_text = f"<SUBJ> {fact_text[0]} <PRED> {fact_text[1]} <OBJ> {fact_text[2]}"
        # Prepare model input
        oie_fact = f"{oie} <FACT> {fact_text}"

        return {
            "oie": oie,
            "fact_input": fact_text,
            "oie_fact": oie_fact,
            "label": ground_truth,
        }


class FactRerankCollater:
    def __init__(self, cfg: CfgNode):
        self.cfg = cfg

    def __call__(self, batch):
        # Prepare output dictionary
        output = {}
        # Transfer what's needed to the batch
        for b in batch:
            for k in b.keys():
                if k not in output:
                    output[k] = []
                output[k].append(b[k])
        # Prepare labels
        output["label"] = torch.tensor(output["label"])

        return output


class UnsupervisedDataset:
    def __init__(self, cfg: CfgNode, mode: str):
        self.cfg = cfg
        assert mode in ["train", "val", "test"]
        dataset_path = (
            self.cfg.TRAIN_DATASET_PATH
            if mode == "train"
            else self.cfg.VAL_DATASET_PATH
            if mode == "val"
            else self.cfg.TEST_DATASET_PATH
        )
        self.dataset = json.load(open(dataset_path))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # Prepare extraction
        e_subj, e_rel, e_obj = self.dataset[idx]["extraction"]
        # Prepare fact
        f_subj, f_rel, f_obj = self.dataset[idx]["fact_ids"]

        return {
            "oie_subj": f"{e_subj}; None",
            "oie_rel": f"{e_rel}; None",
            "oie_obj": f"{e_obj}; None",
            "fact_ids": {"subj": f_subj, "rel": f_rel, "obj": f_obj},
        }


class UnsuperivsedCollater:
    def __init__(self, cfg: CfgNode):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            "princeton-nlp/sup-simcse-bert-base-uncased"
        )

    def __call__(self, batch):
        # Prepare output dictionary
        output = defaultdict(list)
        # Transfer what's needed to the batch
        for b in batch:
            for k in b.keys():
                output[k].append(b[k])
        # Prepare extractions
        for oie in ["subj", "rel", "obj"]:
            output[f"oie_{oie}"] = self.tokenizer(
                output[f"oie_{oie}"],
                padding="max_length",
                add_special_tokens=True,
                return_tensors="pt",
                truncation=True,
                max_length=72,
            )

        return output
