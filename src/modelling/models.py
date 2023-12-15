#          Linking Surface Facts to Large-Scale Knowledge Graphs
# 
#   file: models.py
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
import os
from os.path import join as pjoin
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, RobertaModel, RobertaTokenizerFast
from yacs.config import CfgNode

from utils.setup import get_cfg_defaults


class SlotLinkingModel(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(SlotLinkingModel, self).__init__()
        self.cfg = cfg
        assert self.cfg.MODEL_NAME == "slot-linker"
        self.roberta = RobertaModel.from_pretrained("distilroberta-base")
        # Resizing the embedding matrix and initializing everything else
        self.roberta.resize_token_embeddings(cfg.VOCAB_SIZE)
        self.projector = nn.Linear(768, cfg.EMBED_DIM)
        # As per CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, batch, normalize: bool = True) -> Dict[str, torch.Tensor]:
        embeddings = self.roberta(**batch["oie_input"]).last_hidden_state
        # Perform selection
        selected_embeddings = torch.zeros(
            embeddings.size(0), 3, embeddings.size(-1), device=embeddings.device
        )
        for i, indices in enumerate(batch["oie_pooling_indices"].to(torch.long)):
            selected_embeddings[i] = embeddings[i, indices, ...]
        embeddings = self.projector(selected_embeddings)
        # Normalize embeddings
        if normalize:
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        # subj_obj: [Batch Size, 2, Hidden Size], rel: [Batch Size, 1, Hidden Size]
        return {"subj_obj": embeddings[:, [0, 2], :], "rel": embeddings[:, 1, :]}

    def get_node_embeddings(self, batch, normalize: bool = True) -> torch.Tensor:
        embeddings = self.roberta(**batch["text_input"]).last_hidden_state
        embeddings = self.projector(embeddings[:, 0, :])
        # Normalize embeddings
        if normalize:
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        return embeddings

    def get_entity_embeddings(self, batch, normalize: bool = True) -> torch.Tensor:
        # Get entity embeddings
        entity_embeddings = self.roberta(**batch["entity_text_input"]).last_hidden_state
        entity_embeddings = self.projector(entity_embeddings[:, 0, :])
        # Normalize embeddings
        if normalize:
            entity_embeddings = entity_embeddings / entity_embeddings.norm(
                dim=1, keepdim=True
            )

        return entity_embeddings

    def get_relation_embeddings(self, batch, normalize: bool = True) -> torch.Tensor:
        # Get entity embeddings
        rel_embeddings = self.roberta(**batch["rel_text_input"]).last_hidden_state
        rel_embeddings = self.projector(rel_embeddings[:, 0, :])
        # Normalize embeddings
        if normalize:
            rel_embeddings = rel_embeddings / rel_embeddings.norm(dim=1, keepdim=True)

        return rel_embeddings

    def get_oie_entity_rel_embeddings(
        self, batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        oie_embeddings = self(batch)
        entity_embeddings = self.get_entity_embeddings(batch)
        rel_embeddings = self.get_relation_embeddings(batch)

        return {
            "oie_embeddings": oie_embeddings,
            "entity_embeddings": entity_embeddings,
            "rel_embeddings": rel_embeddings,
        }

    def get_in_batch_text_logits(
        self,
        oie_embeddings: Dict[str, torch.Tensor],
        entity_embeddings: torch.Tensor,
        rel_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Text embeddings
        entity_embeddings = entity_embeddings.transpose(0, 1)
        rel_embeddings = rel_embeddings.transpose(0, 1)
        # Obtain logits
        logit_scale = self.logit_scale.exp()
        subj_logits = (
            logit_scale * oie_embeddings["subj_obj"][:, 0, :] @ entity_embeddings
        )
        obj_logits = (
            logit_scale * oie_embeddings["subj_obj"][:, 1, :] @ entity_embeddings
        )
        rel_logits = logit_scale * oie_embeddings["rel"] @ rel_embeddings

        return {"subj": subj_logits, "obj": obj_logits, "rel": rel_logits}

    def get_in_batch_logits(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        embeddings = self.get_oie_entity_rel_embeddings(batch)
        text_logits = self.get_in_batch_text_logits(
            embeddings["oie_embeddings"],
            embeddings["entity_embeddings"],
            embeddings["rel_embeddings"],
        )

        return text_logits


class FactReranker(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(FactReranker, self).__init__()
        self.cfg = cfg
        assert cfg.MODEL_NAME == "fact-reranker"
        # Prepare tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        special_tokens = ["<SUBJ>", "<PRED>", "<OBJ>", "<FACT>", "<DESC>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        # Prepare model
        self.roberta = RobertaModel.from_pretrained("distilroberta-base")
        self.roberta.resize_token_embeddings(len(self.tokenizer))
        self.classifier = nn.Linear(768, 1)

    def rerank_facts(self, oie: List[str], facts: List[Dict[str, str]]):
        # oie: Batch of OIEs; facts: Batch of set of facts
        # output is same as of the indexer
        slots = ["subj", "rel", "obj"]
        # Prepare outputs
        ids = {slot: [] for slot in slots}
        scores = {slot: [] for slot in slots}
        # Start reranking
        for b in range(len(facts)):
            oie_facts = []
            # Prune sentence from OIE if sentence context is used in the pre-ranker
            if "<SENT>" in oie[b]:
                oie[b] = oie[b].split("<SENT>")[0]
            for f in facts[b]:
                fact = f"<SUBJ> {f['subj']} <PRED> {f['rel']} <OBJ> {f['obj']}"
                oie_facts.append(f"{oie[b]} <FACT> {fact}")
            # Perform reranking
            output = self({"oie_fact": oie_facts})
            ranks = torch.argsort(output["rank_prob"], descending=True)
            reranked_facts = [facts[b][i] for i in ranks]
            # Gather for the batch
            batch_ids = {slot: [] for slot in slots}
            batch_scores = {slot: [] for slot in slots}
            for fact in reranked_facts:
                for slot in slots:
                    batch_ids[slot].append(fact[f"{slot}_id"])
                    batch_scores[slot].append(fact[f"{slot}_score"])
            # Append to output
            for slot in slots:
                ids[slot].append(np.array(batch_ids[slot]))
                scores[slot].append(np.array(batch_scores[slot]))

        return {"ids": ids, "scores": scores}

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Tokenize input
        batch["oie_fact"] = self.tokenizer(
            batch["oie_fact"],
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
        )
        # Move input on device - a bit hacky...
        batch["oie_fact"] = batch["oie_fact"].to(self.roberta.device)
        # Obtain outputs
        features = self.roberta(**batch["oie_fact"]).last_hidden_state[:, 0, :]
        # [Batch Size, 1]
        rank_prob = self.classifier(features)
        # [Batch Size]
        rank_prob = rank_prob.squeeze(-1)

        return {"rank_prob": rank_prob}


class OoKgDetector(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(OoKgDetector, self).__init__()
        assert cfg.MODEL_NAME == "ookg-detector"
        self.cfg = cfg
        # Prepare slot linker
        slot_linker_cfg = get_cfg_defaults()
        slot_linker_cfg.merge_from_file(pjoin(cfg.SLOT_LINKER_PATH, "config.yaml"))
        self.slot_linker = SlotLinkingModel(slot_linker_cfg)
        slot_linker_checkpoint = torch.load(
            pjoin(self.cfg.SLOT_LINKER_PATH, "model_checkpoint.pt"), map_location="cpu"
        )
        self.slot_linker.load_state_dict(slot_linker_checkpoint)
        # Prepare KG index
        self.reset_or_create_kg()
        # Prepare projection layers
        self.query_slot_projectors = nn.ModuleDict(
            {
                key: nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM)
                for key in ["subj", "rel", "obj"]
            }
        )
        self.key_kg_entity_projector = nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM)
        self.key_kg_relation_projector = nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM)
        self.val_kg_entity_projector = nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM)
        self.val_kg_relation_projector = nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM)
        # Initialize projector layers
        for key in self.query_slot_projectors.keys():
            self.query_slot_projectors[key].weight.data = torch.eye(cfg.EMBED_DIM)
        self.key_kg_entity_projector.weight.data = torch.eye(cfg.EMBED_DIM)
        self.val_kg_entity_projector.weight.data = torch.eye(cfg.EMBED_DIM)
        self.key_kg_relation_projector.weight.data = torch.eye(cfg.EMBED_DIM)
        self.val_kg_relation_projector.weight.data = torch.eye(cfg.EMBED_DIM)

    def reset_or_create_kg(self, index_name: str = "kg-index") -> None:
        kg_index_path = pjoin(self.cfg.SLOT_LINKER_PATH, index_name)
        assert os.path.exists(kg_index_path), f"{index_name} has to exist!"
        # Prepare entity data
        entity_embeddings = torch.from_numpy(
            np.load(pjoin(kg_index_path, "entity_embeddings.npy"))
        ).to(self.slot_linker.roberta.device)
        self.entity_embeddings = nn.Parameter(entity_embeddings, requires_grad=False)
        # Prepare relation data
        relation_embeddings = torch.from_numpy(
            np.load(pjoin(kg_index_path, "relation_embeddings.npy"))
        ).to(self.slot_linker.roberta.device)
        self.relation_embeddings = nn.Parameter(
            relation_embeddings, requires_grad=False
        )

    def get_kg(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.entity_embeddings, self.relation_embeddings

    def remove_from_kg_iterative(
        self,
        entities: List[str] = None,
        relations: List[str] = None,
        index_name: str = "kg-index",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert entities or relations, "entities or relations has to provided"
        kg_index_path = pjoin(self.cfg.SLOT_LINKER_PATH, index_name)
        # Reverse entity index
        index2entity = json.load(open(pjoin(kg_index_path, "index2entity.json")))
        # Reverse relation index
        index2rel = json.load(open(pjoin(kg_index_path, "index2rel.json")))
        # Filter entity index
        n_e, h_s = self.entity_embeddings.size()
        entity_embeddings = torch.zeros(
            (n_e - len(entities), h_s), device=self.entity_embeddings.device
        )
        insert_index = 0
        for i, e in tqdm(index2entity.items()):
            if e in entities:
                continue
            entity_embeddings[insert_index] = self.entity_embeddings[int(i)]
            insert_index += 1
        assert insert_index == entity_embeddings.size(
            0
        ), f"Insert index: {insert_index}; Entity embeds: {entity_embeddings.size(0)}"
        # Filter relation index
        n_r, h_s = self.relation_embeddings.size()
        relation_embeddings = torch.zeros(
            (n_r - len(relations), h_s), device=self.relation_embeddings.device
        )
        insert_index = 0
        for i, r in tqdm(index2rel.items()):
            if r in relations:
                continue
            relation_embeddings[insert_index] = self.relation_embeddings[int(i)]
            insert_index += 1
        assert insert_index == relation_embeddings.size(
            0
        ), f"Insert index: {insert_index}; Rel embeds: {relation_embeddings.size(0)}"

        return entity_embeddings, relation_embeddings

    def remove_from_kg_mask_select(
        self,
        entities: List[str] = None,
        relations: List[str] = None,
        index_name: str = "kg-index",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert entities or relations, "entities or relations has to provided"
        kg_index_path = pjoin(self.cfg.SLOT_LINKER_PATH, index_name)
        # Mappings
        index2entity = json.load(open(pjoin(kg_index_path, "index2entity.json")))
        index2rel = json.load(open(pjoin(kg_index_path, "index2rel.json")))
        # Filter entity index
        n_e, h = self.entity_embeddings.size()
        mask = torch.ones(n_e, dtype=torch.bool, device=self.slot_linker.roberta.device)
        remove_entities_indices = [
            int(i) for i, e in index2entity.items() if e in entities
        ]
        mask[remove_entities_indices] = False
        entity_embeddings = torch.masked_select(
            self.entity_embeddings, mask.view(-1, 1)
        ).view(-1, h)
        # Filter relation index
        n_r, h = self.relation_embeddings.size()
        mask = torch.ones(n_r, dtype=torch.bool, device=self.slot_linker.roberta.device)
        remove_relations_indices = [
            int(i) for i, r in index2rel.items() if r in relations
        ]
        mask[remove_relations_indices] = False
        relation_embeddings = torch.masked_select(
            self.relation_embeddings, mask.view(-1, 1)
        ).view(-1, h)

        return entity_embeddings, relation_embeddings

    @torch.no_grad()
    def get_entity_embeddings(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        embeddings = self.slot_linker.get_entity_embeddings(batch, normalize=True)
        return embeddings

    @torch.no_grad()
    def get_relation_embeddings(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        embeddings = self.slot_linker.get_relation_embeddings(batch, normalize=True)
        return embeddings

    @torch.no_grad()
    def get_oie_embeddings(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        embeddings = self.slot_linker(batch, normalize=True)
        # subj_obj: [Batch Size, 2, Hidden Size], rel: [Batch Size, 1, Hidden Size]
        return embeddings

    def kg_attention(
        self, oie_embeddings: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:
        # KG embeddings
        entity_embeddings = kwargs.pop("entity_embeddings", self.entity_embeddings)
        relation_embeddings = kwargs.pop(
            "relation_embeddings", self.relation_embeddings
        )
        # Query-Key-Val projections
        query_oie_embeddings = {
            slot: self.query_slot_projectors[slot](embed)
            for slot, embed in oie_embeddings.items()
        }
        key_entity_embeddings = self.key_kg_entity_projector(entity_embeddings)
        val_entity_embeddings = self.val_kg_entity_projector(entity_embeddings)
        key_relation_embeddings = self.key_kg_relation_projector(relation_embeddings)
        val_relation_embeddings = self.val_kg_relation_projector(relation_embeddings)
        # -------- Subject attention --------
        # [batch_size, hidden_size] @ [hidden_size, num_entities]
        subj_scores = F.softmax(
            query_oie_embeddings["subj"] @ key_entity_embeddings.transpose(0, 1), dim=-1
        )
        # [batch_size, num_entities] @ [num_entities, hidden_size]
        subj_embedding = subj_scores @ val_entity_embeddings
        # -------- Object attention --------
        # [batch_size, hidden_size] @ [hidden_size, num_entities]
        obj_scores = F.softmax(
            query_oie_embeddings["obj"] @ key_entity_embeddings.transpose(0, 1), dim=-1
        )
        # [batch_size, num_entities] @ [num_entities, hidden_size]
        obj_embedding = obj_scores @ val_entity_embeddings
        # -------- Relation attention --------
        # [batch_size, num_relations]
        rel_scores = F.softmax(
            query_oie_embeddings["rel"] @ key_relation_embeddings.transpose(0, 1),
            dim=-1,
        )
        # [batch_size, num_entities] @ [num_entities, hidden_size]
        rel_embedding = rel_scores @ val_relation_embeddings

        return {"subj": subj_embedding, "obj": obj_embedding, "rel": rel_embedding}

    def get_scores(
        self, matchings: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        scores = {}
        predictions = {}
        for slot in matchings.keys():
            scores[slot] = torch.mean(matchings[slot], dim=-1)
            predictions[slot] = matchings[slot].argmax(-1)

        return scores, predictions

    def repack_oie_embeddings(
        self, oie_embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            "subj": oie_embeddings["subj_obj"][:, 0, :],
            "rel": oie_embeddings["rel"],
            "obj": oie_embeddings["subj_obj"][:, 1, :],
        }

    def matching(
        self, oie_embeddings: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:
        entity_embeddings = kwargs.pop("entity_embeddings", self.entity_embeddings)
        relation_embeddings = kwargs.pop(
            "relation_embeddings", self.relation_embeddings
        )
        subj_scores = oie_embeddings["subj"] @ entity_embeddings.transpose(0, 1)
        obj_scores = oie_embeddings["obj"] @ entity_embeddings.transpose(0, 1)
        rel_scores = oie_embeddings["rel"] @ relation_embeddings.transpose(0, 1)

        return {"subj": subj_scores, "obj": obj_scores, "rel": rel_scores}

    def forward(
        self, batch: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Get & repack OIE embeddings
        oie_embeddings = self.repack_oie_embeddings(self.get_oie_embeddings(batch))
        oie_attended_embeddings = self.kg_attention(oie_embeddings, **kwargs)
        # Residual
        oie_embeddings = {
            slot: oie_attended_embeddings[slot] + oie_embeddings[slot]
            for slot in oie_embeddings.keys()
        }
        # Perform matching
        matchings = self.matching(oie_embeddings, **kwargs)
        # Obtain scores
        scores, _ = self.get_scores(matchings)

        return scores

    def get_predictions(
        self, batch: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:
        scores = self(batch, **kwargs)
        predictions = {}
        threshold = kwargs.pop("threshold", 0.5)
        for slot, slot_score in scores.items():
            predictions[slot] = torch.sigmoid(slot_score) > threshold
            # Cast to integers
            predictions[slot] = predictions[slot].to(torch.int32).squeeze(-1)

        return predictions


class UnsupervisedSimCSE(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(UnsupervisedSimCSE, self).__init__()
        self.cfg = cfg
        assert self.cfg.EMBED_DIM == 768, "Wrong embedding size!"
        self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base")

    @torch.no_grad()
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # subj_obj: [Batch Size, 2, Hidden Size], rel: [Batch Size, 1, Hidden Size]
        embeddings = torch.zeros(len(batch["fact_ids"]), 3, self.cfg.EMBED_DIM)
        for i, slot in enumerate(["subj", "rel", "obj"]):
            embeddings[:, i, :] = self.model(**batch[f"oie_{slot}"]).pooler_output

        return {
            "subj_obj": embeddings[:, [0, 2], :].contiguous(),
            "rel": embeddings[:, 1, :].contiguous(),
        }

    @torch.no_grad()
    def get_node_embeddings(self, batch: Dict[str, torch.Tensor]):
        return self.model(**batch["text_input"]).pooler_output
