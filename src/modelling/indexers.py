#          Linking Surface Facts to Large-Scale Knowledge Graphs
# 
#   file: indexers.py
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
import logging
import os
from collections import defaultdict
from os.path import exists as pexists
from os.path import join as pjoin
from typing import Dict, List, Union

import faiss
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, RobertaTokenizerFast
from yacs.config import CfgNode

from utils.wikidata import WikidataProcessed, WikidataZip


class FullWikidata:
    def __init__(
        self, cfg: CfgNode, wikidata: Union[WikidataZip, WikidataProcessed], **kwargs
    ):
        self.cfg = cfg
        self.wikidata = wikidata
        ids_path = kwargs.pop("ids_path", None)
        data = json.load(open(ids_path)) if ids_path else list(self.wikidata.wikidata)
        self.data = [c_id for c_id in data if c_id != "None"]
        logging.info(f"Cleaned {len(data) - len(self.data)} instances...")

    def __len__(self):
        return len(self.data)


class BenchmarkWikidata:
    def __init__(
        self, cfg: CfgNode, wikidata: Union[WikidataZip, WikidataProcessed], **kwargs
    ):
        self.entities, self.rels = set(), set()
        for dataset_name in ["train", "val", "test"]:
            dataset = kwargs.pop(dataset_name, None)
            if not dataset:
                # Just load if not provided
                dataset_name = f"{dataset_name.upper()}_DATASET_PATH"
                dataset = json.load(open(cfg.get(dataset_name)))
            else:
                # Unwrap if provided
                dataset = self.unwrap_dataset(dataset)
            # Gather entities and relations
            for sample in dataset:
                f_subj, f_rel, f_obj = sample["fact_ids"]
                self.entities.add(f_subj)
                self.entities.add(f_obj)
                self.rels.add(f_rel)
        # Convert to lists
        self.data = list(self.entities) + list(self.rels)
        self.wikidata = wikidata

    def __len__(self):
        return len(self.data)

    def unwrap_dataset(self, dataset_object):
        # In case of Subset dataset
        while hasattr(dataset_object, "dataset"):
            dataset_object = dataset_object.dataset
        return dataset_object


class FullWikidataBuilderDataset(Dataset, FullWikidata):
    def __init__(self, cfg: CfgNode, **kwargs):
        super().__init__(cfg, **kwargs)

    def __getitem__(self, idx: int):
        object_id = self.data[idx]
        data = self.wikidata.get_data(object_id)
        text = self.wikidata.get_label(data)
        description = self.wikidata.get_description(data)
        # Prepare special token
        special = "<ENT>" if object_id[0] == "Q" else "<REL>"

        return {"text_input": f"{text} {special} {description}", "id": object_id}


class BenchmarkWikidataBuilderDataset(Dataset, BenchmarkWikidata):
    def __init__(
        self, cfg: CfgNode, wikidata: Union[WikidataZip, WikidataProcessed], **kwargs
    ):
        super().__init__(cfg, wikidata, **kwargs)

    def __getitem__(self, idx: int):
        object_id = self.data[idx]
        data = self.wikidata.get_data(object_id)
        text = self.wikidata.get_label(data)
        description = self.wikidata.get_description(data)
        # Prepare special token
        special = "<ENT>" if object_id[0] == "Q" else "<REL>"

        return {"text_input": f"{text} {special} {description}", "id": object_id}


class UnsupervisedFullWikidataBuilderDataset(Dataset, FullWikidata):
    def __init__(self, cfg: CfgNode, **kwargs):
        super().__init__(cfg, **kwargs)

    def __getitem__(self, idx: int):
        object_id = self.data[idx]
        data = self.wikidata.get_data(object_id)
        text = self.wikidata.get_label(data)
        description = self.wikidata.get_description(data)

        return {"text_input": f"{text}; {description}", "id": object_id}


class UnsupervisedBenchmarkWikidataBuilderDataset(Dataset, BenchmarkWikidata):
    def __init__(
        self, cfg: CfgNode, wikidata: Union[WikidataZip, WikidataProcessed], **kwargs
    ):
        super().__init__(cfg, wikidata, **kwargs)

    def __getitem__(self, idx: int):
        object_id = self.data[idx]
        data = self.wikidata.get_data(object_id)
        text = self.wikidata.get_label(data)
        description = self.wikidata.get_description(data)

        return {"text_input": f"{text}; {description}", "id": object_id}


index_creation_dataset_factory = {
    "benchmark": BenchmarkWikidataBuilderDataset,
    "benchmark_unsupervised": UnsupervisedBenchmarkWikidataBuilderDataset,
    "full_wikidata": FullWikidataBuilderDataset,
    "full_wikidata_unsupervised": UnsupervisedFullWikidataBuilderDataset,
}


class IndexBuilderCollater:
    def __init__(self, cfg: CfgNode):
        if cfg.MODEL_NAME == "slot-linker":
            self.tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
            # Have to add all tokens, so that it's the same tokenizer we used before...
            # somewhat hacky, but what can you do...
            special_tokens = ["<SUBJ>", "<PRED>", "<OBJ>", "<ENT>", "<REL>"]
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens}
            )
        elif cfg.MODEL_NAME == "simcse":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "princeton-nlp/sup-simcse-bert-base-uncased"
            )
        else:
            raise ValueError(f"{cfg.MODEL_NAME} not recognized!")

    def __call__(self, batch):
        # Prepare output dictionary
        output = defaultdict(list)
        # Transfer what's needed to the batch
        for b in batch:
            for k in b.keys():
                output[k].append(b[k])
        # Tokenize entity text
        output["text_input"] = self.tokenizer(
            output["text_input"],
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=32,
        )

        return output


class TextIndexerInterface:
    def __init__(self, cfg):
        # Create index path
        self.index_path = (
            pjoin(cfg.EXPERIMENT_PATH, "kg-index")
            if not cfg.INDEX_PATH
            else cfg.INDEX_PATH
        )

    def search(self, arg: str, features: np.ndarray, n_k: int):
        assert arg in ["entity", "relation"]
        search_methods = {
            "entity": self.faiss_entity_index.search,
            "relation": self.faiss_relation_index.search,
        }
        scores, indices = search_methods[arg](features, n_k)

        return {"scores": scores, "indices": indices}

    def __call__(self, oie_features: Dict[str, torch.Tensor], n_k: int = 100):
        # Convert to numpy
        oie_features = {k: v.cpu().numpy() for k, v in oie_features.items()}
        # [Batch Size, _, Embedding Size]
        b_s, _, e_s = oie_features["subj_obj"].shape
        # Get features
        subj_obj_oie_features = oie_features["subj_obj"].reshape(b_s * 2, e_s)
        rel_oie_features = oie_features["rel"]
        # Get retrieved indices
        retrieved_entities = self.search("entity", subj_obj_oie_features, n_k=n_k)
        retrieved_relations = self.search("relation", rel_oie_features, n_k=n_k)
        # Prepare entity & relation indices & scores (keys)
        retrieved_entities = {
            k: v.reshape(b_s, 2, n_k) for k, v in retrieved_entities.items()
        }
        retrieved_relations = {
            k: v.reshape(b_s, 1, n_k) for k, v in retrieved_relations.items()
        }
        # Prepare output dictionaries
        slots = ["subj", "rel", "obj"]
        ids = {slot: [] for slot in slots}
        scores = {slot: [] for slot in slots}
        # Iterate over the batch size
        for b in range(b_s):
            cur_ids = {arg: [] for arg in slots}
            cur_scores = {arg: [] for arg in slots}
            # Iterate over the retrievals
            for i in range(n_k):
                # Get indices: 0-subject, 1-object, 0-relations
                cur_ids["subj"].append(
                    self.obtain_entity(retrieved_entities["indices"][b][0][i])
                )
                cur_ids["obj"].append(
                    self.obtain_entity(retrieved_entities["indices"][b][1][i])
                )
                cur_ids["rel"].append(
                    self.obtain_relation(retrieved_relations["indices"][b][0][i])
                )
                # Get scores: 0-subject, 1-object, 0-relations
                cur_scores["subj"].append(retrieved_entities["scores"][b][0][i])
                cur_scores["obj"].append(retrieved_entities["scores"][b][1][i])
                cur_scores["rel"].append(retrieved_relations["scores"][b][0][i])
            # Append retrievals & scores for each slot
            for slot in slots:
                ids[slot].append(np.array(cur_ids[slot]))
                scores[slot].append(np.array(cur_scores[slot]))

        return {"ids": ids, "scores": scores}

    def load_entity_index(self):
        self.index2entity = json.load(open(pjoin(self.index_path, "index2entity.json")))
        self.faiss_entity_index = faiss.read_index(
            pjoin(self.index_path, "faiss_index_entity")
        )
        logging.info(f"Total entities in index: {self.faiss_entity_index.ntotal}")

    def load_relation_index(self):
        self.index2relation = json.load(open(pjoin(self.index_path, "index2rel.json")))
        self.faiss_relation_index = faiss.read_index(
            pjoin(self.index_path, "faiss_index_rel")
        )
        logging.info(f"Total relations in index: {self.faiss_relation_index.ntotal}")

    def obtain_entity(self, entity_index):
        return self.index2entity[str(entity_index)] if entity_index > 0 else "None"

    def obtain_relation(self, rel_index):
        return self.index2relation[str(rel_index)] if rel_index > 0 else "None"


class FaissTextIndexCreator(TextIndexerInterface):
    def __init__(self, cfg: CfgNode):
        super(FaissTextIndexCreator, self).__init__(cfg)
        self.cfg = cfg
        self.faiss_entity_index = faiss.index_factory(
            cfg.EMBED_DIM, "Flat", faiss.METRIC_INNER_PRODUCT
        )
        self.faiss_relation_index = faiss.index_factory(
            cfg.EMBED_DIM, "Flat", faiss.METRIC_INNER_PRODUCT
        )
        self.entity_embeddings, self.relation_embeddings = [], []
        self.index2entity, self.index2relation = {}, {}
        self.e_index, self.r_index = 0, 0
        # Check whether index directory exist in the experiment, if not create
        if not pexists(self.index_path):
            os.mkdir(self.index_path)

    def save_entity_index(self):
        # Write entity index & index2entity dictionary
        faiss.write_index(
            self.faiss_entity_index, pjoin(self.index_path, "faiss_index_entity")
        )
        json.dump(
            self.index2entity,
            open(pjoin(self.index_path, "index2entity.json"), "w"),
        )
        np.save(
            pjoin(self.index_path, "entity_embeddings.npy"),
            np.stack(self.entity_embeddings, axis=0),
        )

    def save_relation_index(self):
        # Write entity index & index2entity dictionary
        faiss.write_index(
            self.faiss_relation_index, pjoin(self.index_path, "faiss_index_rel")
        )
        json.dump(
            self.index2relation,
            open(pjoin(self.index_path, "index2rel.json"), "w"),
        )
        np.save(
            pjoin(self.index_path, "relation_embeddings.npy"),
            np.stack(self.relation_embeddings, axis=0),
        )

    def aggregate_embeddings(self, embeddings: torch.Tensor, ids: List[str]):
        embeddings = embeddings.cpu().numpy()
        for i, er_id in enumerate(ids):
            # We're dealing with an entity
            if er_id[0] == "Q":
                self.index2entity[str(self.e_index)] = er_id
                self.e_index += 1
                self.entity_embeddings.append(embeddings[i])
            # We're dealing with a relation (property)
            elif er_id[0] == "P":
                self.index2relation[str(self.r_index)] = er_id
                self.r_index += 1
                self.relation_embeddings.append(embeddings[i])
            # Something has to be wrong
            else:
                raise ValueError(f"Something is wrong: {er_id}")

    def complete_index(self):
        # Add entity embeddings
        self.faiss_entity_index.add(np.stack(self.entity_embeddings, axis=0))
        # Add relation embeddings
        self.faiss_relation_index.add(np.stack(self.relation_embeddings, axis=0))


class FaissTextIndexer(TextIndexerInterface):
    def __init__(self, cfg: CfgNode, **kwargs):
        super(FaissTextIndexer, self).__init__(cfg)
        self.cfg = cfg
        self.load_entity_index()
        self.load_relation_index()
        self.generate_reverse_indices()
        self.wikidata = kwargs.pop("wikidata", None)

    def generate_reverse_indices(self):
        self.entity2index = {e: i for i, e in self.index2entity.items()}
        self.relation2index = {r: i for i, r in self.index2relation.items()}

    def remove_entities_from_index(self, entities: List[str]):
        # Remove from Faiss Index
        indices = np.array(
            [int(self.entity2index[e]) for e in entities if e in self.entity2index]
        )
        self.faiss_entity_index.remove_ids(indices)
        # Update dictionary accordingly
        # Step-1: Delete elements from index2entity
        for index in indices:
            del self.index2entity[str(index)]
        # Step-2: Generate pruned index2entity
        self.index2entity = {
            str(i): e for i, e in enumerate(self.index2entity.values())
        }
        # Step-3: Regenerate entity2index
        self.entity2index = {e: i for i, e in self.index2entity.items()}

    def remove_relations_from_index(self, relations: List[str]):
        indices = np.array(
            [int(self.relation2index[r]) for r in relations if r in self.relation2index]
        )
        self.faiss_relation_index.remove_ids(indices)
        # Update dictionary accordingly
        # Step-1: Delete elements from index2relation
        for index in indices:
            del self.index2relation[str(index)]
        # Step-2: Generate pruned index2entity
        self.index2relation = {
            str(i): e for i, e in enumerate(self.index2relation.values())
        }
        # Step-3: Regenerate entity2index
        self.relation2index = {e: i for i, e in self.index2relation.items()}

    def get_wiki_desc(self, object_id: str):
        # If indexer returned None, return None :)
        if object_id == "None":
            return "None"
        # Fetch data & label and return label
        data = self.wikidata.get_data(object_id)
        desc = self.wikidata.get_description(data)

        return desc

    def get_wiki_label(self, object_id: str):
        # If indexer returned None, return None :)
        if object_id == "None":
            return "None"
        # Fetch data & label and return label
        data = self.wikidata.get_data(object_id)
        label = self.wikidata.get_label(data)

        return label

    def get_top_k_facts(self, retrievals, k: int = 3):
        assert self.wikidata, "Wikidata is necessary for this..."
        # All facts
        facts = []
        for b in range(len(retrievals["ids"]["subj"])):
            # All facts for each element in the batch
            b_facts = []
            for s in range(k):
                for r in range(k):
                    for o in range(k):
                        indices = {"subj": s, "rel": r, "obj": o}
                        fact = {}
                        for slot in retrievals["ids"].keys():
                            # Get id
                            e_id = retrievals["ids"][slot][b][indices[slot]]
                            fact[f"{slot}_id"] = e_id
                            # Get label & description
                            label = self.get_wiki_label(e_id)
                            desc = self.get_wiki_desc(e_id)
                            fact[slot] = f"{label} <DESC> {desc}"
                            # Get score
                            score = retrievals["scores"][slot][b][indices[slot]]
                            fact[f"{slot}_score"] = score
                        b_facts.append(fact)
            # Append all facts for the batch element
            facts.append(b_facts)

        return facts

    def get_top_k_facts_wrt_rel(self, retrievals, k: int = 100):
        assert self.wikidata, "Wikidata is necessary for this..."
        # All facts
        facts = []
        for b in range(len(retrievals["ids"]["subj"])):
            # All facts for each element in the batch
            b_facts = []
            for r in range(k):
                indices = {"subj": 0, "rel": r, "obj": 0}
                fact = {}
                for slot in retrievals["ids"].keys():
                    # Get id
                    e_id = retrievals["ids"][slot][b][indices[slot]]
                    fact[f"{slot}_id"] = e_id
                    # Get label & description
                    label = self.get_wiki_label(e_id)
                    desc = self.get_wiki_desc(e_id)
                    fact[slot] = f"{label} <DESC> {desc}"
                    # Get score
                    score = retrievals["scores"][slot][b][indices[slot]]
                    fact[f"{slot}_score"] = score
                b_facts.append(fact)
            # Append all facts for the batch element
            facts.append(b_facts)

        return facts


class FaissGraphIndexer:
    def __init__(self, cfg: CfgNode):
        self.index2entity = json.load(
            open(pjoin(cfg.FAISS_INDEX_PATH, "index2entity.json"))
        )
        self.faiss_entity_index = faiss.read_index(
            pjoin(cfg.FAISS_INDEX_PATH, "faiss_index_entity")
        )
        self.index2relation = json.load(
            open(pjoin(cfg.FAISS_INDEX_PATH, "index2relation.json"))
        )
        self.faiss_relation_index = faiss.read_index(
            pjoin(cfg.FAISS_INDEX_PATH, "faiss_index_relation")
        )
        try:
            faiss.ParameterSpace().set_index_parameter(
                self.faiss_entity_index, "nprobe", 1024
            )
        except RuntimeError:
            print("Cound not set nprobes, deterministic index!")

    def __call__(self, oie_features: Dict[str, torch.Tensor]):
        # [Batch Size, _, Embedding Size]
        b_s, _, e_s = oie_features["subj_obj"].size()
        # Get features
        subj_obj_oie_features = oie_features["subj_obj"].view(b_s * 2, e_s)
        rel_oie_features = oie_features["rel"]
        # Convert to numpy
        subj_obj_oie_features = subj_obj_oie_features.cpu().numpy()
        rel_oie_features = rel_oie_features.cpu().numpy()
        # Get retrieved indices
        _, all_retrieved_entity_indices = self.faiss_entity_index.search(
            subj_obj_oie_features, 100
        )
        _, all_retrieved_relation_indices = self.faiss_relation_index.search(
            rel_oie_features, 100
        )
        # [Batch Size, 2, K = 100]
        all_retrieved_entity_indices = all_retrieved_entity_indices.reshape(b_s, 2, 100)
        all_retrieved_relation_indices = all_retrieved_relation_indices.reshape(
            b_s, 1, 100
        )
        retrievals = {"subj": [], "obj": [], "rel": []}
        # Gather all retrieved entities
        for b in range(b_s):
            # Get retrieved entities
            subj_retrievals = np.array(
                [
                    self.index2entity[str(entity_index)] if entity_index > 0 else "None"
                    for entity_index in all_retrieved_entity_indices[b][0]
                ]
            )
            obj_retrievals = np.array(
                [
                    self.index2entity[str(entity_index)] if entity_index > 0 else "None"
                    for entity_index in all_retrieved_entity_indices[b][1]
                ]
            )
            rel_retrievals = np.array(
                [
                    self.index2relation[str(rel_index)] if rel_index > 0 else "None"
                    for rel_index in all_retrieved_relation_indices[b][0]
                ]
            )
            retrievals["subj"].append(subj_retrievals)
            retrievals["obj"].append(obj_retrievals)
            retrievals["rel"].append(rel_retrievals)

        return retrievals


index_factory = {"graph": FaissGraphIndexer, "text": FaissTextIndexer}
