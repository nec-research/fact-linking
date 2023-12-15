#          Linking Surface Facts to Large-Scale Knowledge Graphs
# 
#   file: evaluators.py
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

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from yacs.config import CfgNode


class LinkingEvaluator:
    def __init__(self, num_samples: int, eval_types: List[str]):
        self.num_samples = num_samples
        self.eval_types = eval_types
        self.name2metric = {
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "accuracy": accuracy_score,
        }
        self.best_result = -1.0
        self.reset()

    def reset(self):
        self.index = 0
        self.preds = {name: np.zeros(self.num_samples) for name in self.eval_types}
        self.gts = {name: np.zeros(self.num_samples) for name in self.eval_types}

    def evaluate(self):
        metrics = defaultdict(dict)
        for e in self.eval_types:
            for metric_name in self.name2metric.keys():
                # Gorjan: Bit of a hack, but to save time
                if metric_name == "accuracy":
                    metric_score = self.name2metric[metric_name](
                        y_true=self.gts[e], y_pred=self.preds[e]
                    )
                else:
                    metric_score = self.name2metric[metric_name](
                        y_true=self.gts[e], y_pred=self.preds[e], average="micro"
                    )

            metrics[e][metric_name] = metric_score

        return metrics

    def is_best(self):
        # Finding best model based on accuracy score always
        metrics = self.evaluate()
        # Find average score based on all evalution options (subj., obj., rel., etc.)
        avg_score = sum([metrics[e]["accuracy"] for e in self.eval_types]) / len(
            self.eval_types
        )
        if avg_score > self.best_result:
            self.best_result = avg_score
            return True
        return False

    def process(self, model_output: Dict[str, torch.Tensor], batch: Dict[str, Any]):
        for e in self.eval_types:
            output = model_output[e].argmax(-1)
            batch_size = output.size(0)
            gt = batch[f"{e}-index"]
            self.preds[e][self.index : self.index + batch_size] = output.cpu().numpy()
            self.gts[e][self.index : self.index + batch_size] = gt.cpu().numpy()
        self.index += batch_size


class RetrievalEvaluator:
    def __init__(self, num_samples: int, cfg: CfgNode):
        self.num_samples = num_samples
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.embeddings = {
            k: torch.zeros((self.num_samples, self.cfg.EMBED_DIM))
            for k in ["fact", "oie"]
        }
        self.fact2index = {}
        self.reverse_index = {}
        self.index = 0
        self.r_index = 0
        self.facts = []
        self.last_fact_saw = None

    def process(self, model_output: Dict[str, torch.Tensor], facts: List[str]):
        for embeddings_name in model_output.keys():
            batch_size = model_output[embeddings_name].size(0)
            self.embeddings[embeddings_name][
                self.index : self.index + batch_size
            ] = model_output[embeddings_name].cpu()
        # Update indices
        if self.last_fact_saw and facts[0] != self.last_fact_saw:
            self.r_index += 1
        for i in range(batch_size):
            self.reverse_index[self.index] = self.r_index
            if facts[i] not in self.fact2index:
                self.fact2index[facts[i]] = self.index
            # Check for reverse index
            if i + 1 < batch_size:
                if facts[i] != facts[i + 1]:
                    self.r_index += 1
            self.facts.append(facts[i])
            self.index += 1
        # Save last fact to look across batches
        self.last_fact_saw = facts[-1]

    def evaluate(self):
        # print(self.fact2index)
        recalls = {"top1": 0, "top5": 0, "top10": 0, "top100": 0}
        # Get only unique fact embeddings
        unique_fact_indices = torch.tensor(list(self.fact2index.values()))
        self.embeddings["fact"] = self.embeddings["fact"][unique_fact_indices]
        # Transfer embeddings back on the desired device
        for key in self.embeddings.keys():
            self.embeddings[key] = self.embeddings[key].to(self.cfg.DEVICE)
        # Transpose the fact embeddings once
        self.embeddings["fact"] = self.embeddings["fact"].transpose(0, 1)
        # Compute the scores
        for i in tqdm(range(self.embeddings["oie"].size(0))):
            # Get scores [1, embed_dim] @ [embed_dim, total_samples]
            scores = self.embeddings["oie"][i] @ self.embeddings["fact"]
            scores = scores.cpu().numpy()
            # Get prediction indices and ground truth index
            pred_indices = scores.argsort()[::-1]
            gt_index = np.array([self.reverse_index[i]])
            # Compute the recalls
            for j in [1, 5, 10, 100]:
                if (pred_indices[:j] == gt_index).sum():
                    recalls[f"top{j}"] += 1
        # Normalize to obtain recalls
        for k in recalls.keys():
            recalls[k] = np.round(recalls[k] / self.num_samples * 100, decimals=2)

        return recalls


class SlotLinkingEvaluator:
    # Assuming contrastive setup, model outputs are top-k predictions, i.e., retrievals
    def __init__(self, cfg: CfgNode, **kwargs):
        self.cfg = cfg
        self.best_result = -1.0
        self.reset()

    def reset(self):
        self.results = {}
        for k in ["1", "5", "10", "100"]:
            if k not in self.results:
                self.results[k] = {}
            for slot in ["subj", "rel", "obj", "fact"]:
                if slot not in self.results[k]:
                    self.results[k][slot] = []
        self.total = 0
        self.evaluated = False

    def is_best(self):
        assert self.evaluated, "Have to execute self.evaluate first"
        # Measure total recall
        total = 0.0
        for k in self.results.keys():
            for slot in self.results[k].keys():
                recall = self.results[k][slot]["recall"]
                total += recall
        # Check if bst
        if total >= self.best_result:
            self.best_result = total
            return True

        return False

    def process(self, retrievals, fact_ids: List[Dict[str, str]]):
        b_s = len(retrievals["subj"])
        for b in range(b_s):
            for k in self.results.keys():
                # Count correct_slots, if 3, increase fact score
                correct_slots = 0
                for slot in retrievals.keys():
                    predictions = retrievals[slot][b]
                    if (predictions[: int(k)] == fact_ids[b][slot]).sum():
                        self.results[k][slot].append(1)
                        correct_slots += 1
                    else:
                        self.results[k][slot].append(0)
                if correct_slots == 3:
                    self.results[k]["fact"].append(1)
                else:
                    self.results[k]["fact"].append(0)
        # Update totals
        self.total += b_s

    def evaluate(self):
        for k in self.results.keys():
            for slot in self.results[k]:
                recall = np.round(
                    np.sum(self.results[k][slot]) / self.total * 100.0, decimals=2
                )
                error = np.round(
                    np.std(self.results[k][slot], ddof=1) / np.sqrt(self.total) * 100.0,
                    decimals=2,
                )
                self.results[k][slot] = {"recall": recall, "error": error}
        # Set evaluation status to true
        self.evaluated = True

        return self.results


class SlotLinkingPerRelationEvaluator:
    # Assuming contrastive setup, model outputs are top-k predictions, i.e., retrievals
    def __init__(self, cfg: CfgNode, **kwargs):
        self.cfg = cfg
        # Gather relations
        self.relations = set()
        eval_loader = kwargs.pop("eval_loader", None)
        assert eval_loader, "This evaluator requires the evaluation dataset as input"
        eval_dataset = self.unwrap_dataset(eval_loader)
        for sample in eval_dataset:
            _, rel, _ = sample["fact_ids"]
            self.relations.add(rel)
        # Prepare meters
        self.reset()

    @staticmethod
    def unwrap_dataset(dataset_object):
        # Find dataset inside dataloader, subset, dataset
        while hasattr(dataset_object, "dataset"):
            dataset_object = dataset_object.dataset
        return dataset_object

    def reset(self):
        self.results = {}
        for rel in self.relations:
            if rel not in self.results:
                self.results[rel] = {}
            for k in ["1", "5", "10", "100"]:
                if k not in self.results[rel]:
                    self.results[rel][k] = {}
                for slot in ["subj", "rel", "obj", "fact"]:
                    if slot not in self.results[rel][k]:
                        self.results[rel][k][slot] = []

    def process(self, retrievals, fact_ids: List[Dict[str, str]]):
        for b in range(len(retrievals["subj"])):
            rel = fact_ids[b]["rel"]
            for k in self.results[rel].keys():
                # Count correct_slots, if 3, increase fact score
                correct_slots = 0
                for slot in retrievals.keys():
                    predictions = retrievals[slot][b]
                    if (predictions[: int(k)] == fact_ids[b][slot]).sum():
                        self.results[rel][k][slot].append(1)
                        correct_slots += 1
                    else:
                        self.results[rel][k][slot].append(0)
                if correct_slots == 3:
                    self.results[rel][k]["fact"].append(1)
                else:
                    self.results[rel][k]["fact"].append(0)

    def evaluate(self):
        for rel in self.results.keys():
            for k in self.results[rel].keys():
                for slot in self.results[rel][k].keys():
                    total = len(self.results[rel][k][slot])
                    recall = np.round(
                        np.sum(self.results[rel][k][slot]) / total * 100.0,
                        decimals=2,
                    )
                    error = np.round(
                        np.std(self.results[rel][k][slot], ddof=1)
                        / np.sqrt(total)
                        * 100.0,
                        decimals=2,
                    )
                    self.results[rel][k][slot] = {"recall": recall, "error": error}

        return self.results


evaluators_factory = {
    "regular": SlotLinkingEvaluator,
    "per-relation": SlotLinkingPerRelationEvaluator,
}


class KgPresenceEvaluator:
    def __init__(self, cfg: CfgNode):
        self.cfg = cfg
        # gt=1, status=inside; gt=0, status=outside
        self.reseted = False
        self.results = {}
        self.macro_ready = False

    def reset(self, status: str):
        self.reseted = True
        self.status = status
        self.results[self.status] = {"subj": [], "rel": [], "obj": [], "fact": []}

    def process(self, kg_presence_output):
        assert self.reseted, "Evaluator has to be reset with new status before using"
        for slot in kg_presence_output.keys():
            # Casting and decoupling
            if isinstance(kg_presence_output[slot], torch.Tensor):
                preds = kg_presence_output[slot].cpu().numpy()
            elif isinstance(kg_presence_output, dict):
                preds = np.array(kg_presence_output[slot]["decisions"])
            ground_truth = 1 if self.status == "inside" else 0
            self.results[self.status][slot].extend((preds == ground_truth).tolist())
        # Fact results
        fact_results = np.zeros(len(self.results[self.status]["subj"]))
        for slot in ["subj", "rel", "obj"]:
            fact_results += np.array(self.results[self.status][slot])
        fact_results = (fact_results == 3).astype(np.int32)
        self.results[self.status]["fact"].extend(fact_results.tolist())

    def evaluate(self):
        # Normalize scores & return results
        for slot in self.results[self.status].keys():
            total = len(self.results[self.status][slot])
            acc = np.round(
                np.sum(self.results[self.status][slot]) / total * 100.0, decimals=2
            )
            error = np.round(
                np.std(self.results[self.status][slot], ddof=1)
                / np.sqrt(total)
                * 100.0,
                decimals=2,
            )
            self.results[self.status][slot] = {"accuracy": acc, "error": error}
        # Unreset evaluator
        self.reseted = False
        self.macro_ready = True

        return self.results[self.status]

    def evaluate_macro(self):
        assert len(self.results.keys()) == 2, "Both inside and outside required"
        assert self.macro_ready, "Evaluate has to be called for both inside/outside"
        # for status in self.results.keys():
        # for slot in self.results[status].keys():
        macro = {}
        for slot in self.results[self.status].keys():
            slot_total_accuracy = 0
            slot_total_error = 0
            for status in self.results.keys():
                slot_total_accuracy += self.results[status][slot]["accuracy"]
                slot_total_error += self.results[status][slot]["error"]
            macro[slot] = {
                "accuracy": np.round(slot_total_accuracy / 2, 1),
                "error": np.round(slot_total_error / 2, 1),
            }

        return macro
