#          Linking Surface Facts to Large-Scale Knowledge Graphs
# 
#   file: train-slot-linking.py
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
import logging
import os
import random
from os.path import join as pjoin
from typing import Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from yacs.config import CfgNode

from modelling.datasets import BaseCollater, BaseDataset, SlotLinkingTrainingCollater
from modelling.evaluators import SlotLinkingEvaluator
from modelling.indexers import (
    BenchmarkWikidataBuilderDataset,
    FaissTextIndexCreator,
    IndexBuilderCollater,
)
from modelling.models import SlotLinkingModel
from utils.misc import move_batch_to_device
from utils.setup import get_cfg_defaults
from utils.wikidata import wikidata_factory


class LabelGenerator:
    """Generate labels for the slot linking task."""

    def __call__(self, batch, device):
        output = {"subj": [], "rel": [], "obj": []}
        for fact_ids in batch["fact_ids"]:
            output["subj"].append(batch["entity2index"][fact_ids["subj"]])
            output["rel"].append(batch["rel2index"][fact_ids["rel"]])
            output["obj"].append(batch["entity2index"][fact_ids["obj"]])
        # Convert to tensors
        output = {k: torch.tensor(v, device=device) for k, v in output.items()}

        return output


class SlotLinkingCriterion(nn.Module):
    """Slot linking criterion for training the model."""

    def __init__(self, cfg: CfgNode):
        super(SlotLinkingCriterion, self).__init__()
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self, all_logits: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]
    ):
        total_loss = 0
        for slot in all_logits.keys():
            total_loss += self.criterion(all_logits[slot], labels[slot])

        return total_loss / len(all_logits.keys())


def prepare(cfg: CfgNode):
    if cfg.LOG_TO_FILE:
        logging.basicConfig(
            level=logging.INFO,
            filename=pjoin(cfg.EXPERIMENT_PATH, "experiment_log.log"),
            filemode="w",
        )
    else:
        logging.basicConfig(level=logging.INFO)
    device = torch.device(cfg.DEVICE)
    # Load Wikidata
    logging.info("Loading Wikidata...")
    wikidata = wikidata_factory[cfg.WIKIDATA_TYPE](wikidata_path=cfg.WIKIDATA_PATH)
    logging.info("Preparing datasets...")
    train_collater = SlotLinkingTrainingCollater(cfg, wikidata=wikidata)
    cfg.VOCAB_SIZE = len(train_collater.tokenizer)
    # Preparing train dataset
    train_dataset = BaseDataset(cfg, mode="train")
    # Subset the train dataset
    if cfg.TRAIN_SUBSET:
        train_indices = random.sample(range(len(train_dataset)), k=cfg.TRAIN_SUBSET)
        train_dataset = Subset(train_dataset, train_indices)
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Graph negative entities: {cfg.GRAPH_NEGATIVE_ENTITIES}")
    logging.info(f"Graph negative relations: {cfg.GRAPH_NEGATIVE_RELATIONS}")
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=train_collater,
    )
    # Preparing validation dataset
    val_dataset = BaseDataset(cfg, mode="val")
    inference_collater = BaseCollater(cfg)
    if cfg.VAL_SUBSET:
        val_indices = random.sample(range(len(val_dataset)), k=cfg.VAL_SUBSET)
        val_dataset = Subset(val_dataset, val_indices)
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=inference_collater,
    )
    # Indexer dataset
    logging.info("Preparing indexer data...")
    indexer_dataset = BenchmarkWikidataBuilderDataset(
        cfg=cfg, wikidata=wikidata, train=train_dataset, val=val_dataset
    )
    indexer_loader = DataLoader(
        indexer_dataset,
        shuffle=False,
        batch_size=cfg.BATCH_SIZE,
        collate_fn=IndexBuilderCollater(cfg),
        num_workers=cfg.NUM_WORKERS,
    )

    return cfg, train_loader, val_loader, indexer_loader, device


def train(
    cfg: CfgNode,
    train_loader: DataLoader,
    val_loader: DataLoader,
    indexer_loader: DataLoader,
    device,
):
    # Preparing model, loss, optimizer, etc.
    model = SlotLinkingModel(cfg).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    logging.info("Preparing loss, indexer and evaluator...")
    label_generator = LabelGenerator()
    criterion = SlotLinkingCriterion(cfg)
    evaluator = SlotLinkingEvaluator(cfg)
    logging.info("Starting training...")
    for epoch in range(cfg.EPOCHS):
        # Training
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                batch = move_batch_to_device(batch, device)
                # Remove past gradients
                optimizer.zero_grad()
                # Obtain logits
                logits = model.get_in_batch_logits(batch)
                # Obtain labels
                labels = label_generator(batch, device)
                # Measure loss & update weights
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})
        # Prepare model for index update & evaluation
        model.train(False)
        # Update indexer
        logging.info("Creating Faiss index...")
        indexer = FaissTextIndexCreator(cfg)
        for batch in tqdm(indexer_loader):
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                embedding = model.get_node_embeddings(batch)
            # Aggregate embeddings
            indexer.aggregate_embeddings(embeddings=embedding, ids=batch["id"])
        logging.info("Completing index...")
        indexer.complete_index()
        # Validation
        logging.info("Starting validation...")
        evaluator.reset()
        for batch in tqdm(val_loader):
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                oie_features = model(batch)
                grounding = indexer(oie_features)
            evaluator.process(retrievals=grounding["ids"], fact_ids=batch["fact_ids"])
        # Metrics
        logging.info(f"================= Metrics at epoch {epoch+1} =================")
        metrics = evaluator.evaluate()
        for k in metrics.keys():
            logging.info(f"============ Recalls@{k} ============")
            for slot in metrics[k].keys():
                recall = metrics[k][slot]["recall"]
                error = metrics[k][slot]["error"]
                logging.info(f"{slot} = {recall} +/- {error}")
        # Find best model
        if evaluator.is_best():
            logging.info("Found new best! Saving model & entity index...")
            torch.save(
                model.state_dict(), pjoin(cfg.EXPERIMENT_PATH, "model_checkpoint.pt")
            )
            indexer.save_entity_index()
            indexer.save_relation_index()


def main():
    parser = argparse.ArgumentParser(description="Basic training script.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the experiment config path",
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    cfg.CONFIG_PATH = args.config_path
    if args.opts:
        cfg.merge_from_list(args.opts)
    # Create experiment directory
    os.makedirs(cfg.EXPERIMENT_PATH, exist_ok=False)
    # Prepare data, devices, etc.
    cfg, train_loader, val_loader, indexer_loader, device = prepare(cfg)
    cfg.freeze()
    # Save config
    with open(os.path.join(cfg.EXPERIMENT_PATH, "config.yaml"), "w") as f:
        f.write(cfg.dump())
    # Start training
    train(cfg, train_loader, val_loader, indexer_loader, device)


if __name__ == "__main__":
    main()
