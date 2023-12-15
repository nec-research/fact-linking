#          Linking Surface Facts to Large-Scale Knowledge Graphs
# 
#   file: inference-slot-linking.py
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

import argparse
import json
import logging
import random
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from yacs.config import CfgNode

from modelling.datasets import (
    BaseCollater,
    BaseDataset,
    UnsuperivsedCollater,
    UnsupervisedDataset,
)
from modelling.evaluators import evaluators_factory
from modelling.indexers import FaissTextIndexer
from modelling.models import FactReranker, SlotLinkingModel, UnsupervisedSimCSE
from utils.misc import move_batch_to_device
from utils.setup import get_cfg_defaults
from utils.wikidata import Wikidata, wikidata_factory


def prepare(cfg: CfgNode):
    logging.basicConfig(level=logging.INFO)
    device = torch.device(cfg.DEVICE)
    logging.info("=" * 40)
    logging.info(cfg)
    logging.info("=" * 40)
    logging.info("Preparing datasets...")
    # Preparing testing dataset & collater based on the name of the model
    if cfg.MODEL_NAME == "slot-linker":
        test_dataset = BaseDataset(cfg, mode="test")
        inference_collater = BaseCollater(cfg)
    elif cfg.MODEL_NAME == "simcse":
        test_dataset = UnsupervisedDataset(cfg, mode="test")
        inference_collater = UnsuperivsedCollater(cfg)
    if cfg.TEST_SUBSET:
        test_indices = random.sample(range(len(test_dataset)), k=cfg.TEST_SUBSET)
        test_dataset = Subset(test_dataset, test_indices)
    logging.info(f"Testing dataset size: {len(test_dataset)}")
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=inference_collater,
    )
    logging.info("Preparing Wikidata...")
    wikidata = wikidata_factory[cfg.WIKIDATA_TYPE](wikidata_path=cfg.WIKIDATA_PATH)

    return cfg, test_loader, wikidata, device


def prepare_fact_reranker(args, device):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(pjoin(args.fact_reranking_experiment_path, "config.yaml"))
    model = FactReranker(cfg)
    checkpoint = torch.load(
        pjoin(cfg.EXPERIMENT_PATH, "model_checkpoint.pt"), map_location="cpu"
    )
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.train(False)

    return model


def prepare_slot_linker(cfg, device):
    model = SlotLinkingModel(cfg)
    checkpoint = torch.load(
        pjoin(cfg.EXPERIMENT_PATH, "model_checkpoint.pt"), map_location="cpu"
    )
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.train(False)

    return model


def prepare_simcse(cfg, device):
    model = UnsupervisedSimCSE(cfg)
    model.train(False)
    model = model.to(device)

    return model


@torch.no_grad()
def inference(cfg: CfgNode, test_loader: DataLoader, wikidata: Wikidata, device, args):
    if cfg.MODEL_NAME == "slot-linker":
        slot_linker = prepare_slot_linker(cfg, device)
    elif cfg.MODEL_NAME == "simcse":
        slot_linker = prepare_simcse(cfg, device)
    else:
        raise ValueError(f"Unknown model name: {cfg.MODEL_NAME}")
    logging.info("Preparing indexer")
    indexer = FaissTextIndexer(cfg, wikidata=wikidata)
    logging.info("Preparing evaluator...")
    slot_linking_evaluator = evaluators_factory[args.evaluator_type](
        cfg, eval_loader=test_loader
    )
    # Fact reranking
    if args.fact_reranking_experiment_path:
        fact_reranker = prepare_fact_reranker(args, device)
    # Inference
    slot_linking_evaluator.reset()
    logging.info("Starting inference...")
    for batch in tqdm(test_loader):
        batch = move_batch_to_device(batch, device)
        # Obtain outputs
        oie_features = slot_linker(batch)
        # Perform ranking with bi-encoder
        grounding = indexer(oie_features, n_k=100)
        # Perform reranking
        if args.fact_reranking_experiment_path:
            facts = indexer.get_top_k_facts(grounding, k=args.reranker_k)
            grounding = fact_reranker.rerank_facts(batch["oie"], facts)
        # Process evaluators
        slot_linking_evaluator.process(
            retrievals=grounding["ids"], fact_ids=batch["fact_ids"]
        )
    # Grounding/Linking metrics
    metrics = slot_linking_evaluator.evaluate()
    if args.evaluator_type == "regular":
        for k in metrics.keys():
            logging.info(f"========= Recalls@{k} ===========")
            for slot in metrics[k].keys():
                recall = metrics[k][slot]["recall"]
                error = metrics[k][slot]["error"]
                logging.info(f"{slot} = {recall} +/- {error}")
    elif args.evaluator_type == "per-relation":
        for rel_id in metrics.keys():
            rel_data = wikidata.get_data(rel_id)
            rel_label = wikidata.get_label(rel_data)
            logging.info(f"========= {rel_id} - {rel_label} ===========")
            for slot in metrics[rel_id]["1"].keys():
                recall = metrics[rel_id]["1"][slot]["recall"]
                error = metrics[rel_id]["1"][slot]["error"]
                logging.info(f"{slot} = {recall} +/- {error}")
        json.dump(metrics, open(pjoin(cfg.EXPERIMENT_PATH, "metrics.json"), "w"))


def main():
    parser = argparse.ArgumentParser(description="Slot-linking inference.")
    parser.add_argument(
        "--slot_linking_experiment_path",
        type=str,
        required=True,
        help="Path to the slot-linking experiment.",
    )
    parser.add_argument(
        "--fact_reranking_experiment_path",
        type=str,
        default=None,
        help="Path to the fact-reranking experiment.",
    )
    parser.add_argument("--evaluator_type", type=str, default="regular")
    parser.add_argument("--reranker_k", type=int, default=3)
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(pjoin(args.slot_linking_experiment_path, "config.yaml"))
    if args.opts:
        cfg.merge_from_list(args.opts)
    # Prepare data, devices, etc.
    cfg, test_loader, wikidata, device = prepare(cfg)
    # No indexer is updated during inference
    cfg.freeze()
    # Start training
    inference(cfg, test_loader, wikidata, device, args)


if __name__ == "__main__":
    main()
