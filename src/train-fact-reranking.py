#          Linking Surface Facts to Large-Scale Knowledge Graphs
# 
#   file: train-fact-reranking.py
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

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from yacs.config import CfgNode

from modelling.datasets import FactRerankCollater, FactRerankDataset
from modelling.models import FactReranker
from utils.setup import get_cfg_defaults
from utils.wikidata import wikidata_factory


def prepare(cfg: CfgNode, accelerator: Accelerator):
    if cfg.LOG_TO_FILE:
        if accelerator.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                filename=pjoin(cfg.EXPERIMENT_PATH, "experiment_log.log"),
                filemode="w",
            )
    else:
        logging.basicConfig(level=logging.INFO)
    # Load Wikidata
    if accelerator.is_main_process:
        logging.info("Loading Wikidata...")
    wikidata = wikidata_factory[cfg.WIKIDATA_TYPE](wikidata_path=cfg.WIKIDATA_PATH)
    if accelerator.is_main_process:
        logging.info("Preparing datasets...")
    collater = FactRerankCollater(cfg)
    # Preparing train dataset
    train_dataset = FactRerankDataset(cfg, mode="train", wikidata=wikidata)
    # Subset the train dataset
    if cfg.TRAIN_SUBSET:
        train_indices = random.sample(range(len(train_dataset)), k=cfg.TRAIN_SUBSET)
        train_dataset = Subset(train_dataset, train_indices)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collater,
    )
    # Preparing validation dataset
    val_dataset = FactRerankDataset(cfg, mode="val", wikidata=wikidata)
    if cfg.VAL_SUBSET:
        val_indices = random.sample(range(len(val_dataset)), k=cfg.VAL_SUBSET)
        val_dataset = Subset(val_dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collater,
    )
    if accelerator.is_main_process:
        logging.info(f"Train dataset size: {len(train_dataset)}")
        logging.info(f"Validation dataset size: {len(val_dataset)}")

    return cfg, train_loader, val_loader


def train(
    cfg: CfgNode,
    train_loader: DataLoader,
    val_loader: DataLoader,
    accelerator: Accelerator,
):
    # Preparing model, loss, optimizer, etc.
    model = FactReranker(cfg)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    if accelerator.is_main_process:
        logging.info("Starting training...")
    best_result = -1.0
    # Accelerate
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    for epoch in range(cfg.EPOCHS):
        # Training
        model.train(True)
        with tqdm(
            total=len(train_loader), disable=not accelerator.is_main_process
        ) as pbar:
            for batch in train_loader:
                # Remove past gradients
                optimizer.zero_grad()
                # Obtain logits
                output = model(batch)
                # Measure loss & update weights
                loss = criterion(output["rank_prob"], batch["label"])
                accelerator.backward(loss)
                optimizer.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})
        # Validation
        model.train(False)
        metrics = {"correct": 0, "total": 0}
        for batch in tqdm(val_loader, disable=not accelerator.is_main_process):
            with torch.no_grad():
                model_output = model(batch)
            # Gather
            outputs = accelerator.gather(model_output)
            labels = accelerator.gather(batch["label"])
            # Metrics measure
            logits = torch.sigmoid(outputs["rank_prob"])
            metrics["correct"] += ((logits > 0.5) == labels).sum().item()
            metrics["total"] += logits.size(0)
        # Evaluate & save model
        accelerator.wait_for_everyone()
        # Metrics report
        if accelerator.is_main_process:
            logging.info(
                f"================= Metrics at epoch {epoch+1} ================="
            )
            accuracy = round(metrics["correct"] / metrics["total"], 5)
            logging.info(f"Accuracy: {accuracy}")
            # Find best model
            if accuracy > best_result:
                best_result = accuracy
                logging.info("Found new best! Saving model...")
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(
                    unwrapped_model.state_dict(),
                    pjoin(cfg.EXPERIMENT_PATH, "model_checkpoint.pt"),
                )


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
    accelerator = Accelerator(
        cpu=cfg.DEVICE == "cpu",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.is_main_process:
        os.makedirs(cfg.EXPERIMENT_PATH, exist_ok=False)
        # Save config
        with open(os.path.join(cfg.EXPERIMENT_PATH, "config.yaml"), "w") as f:
            f.write(cfg.dump())
    # Prepare data, devices, etc.
    cfg, train_loader, val_loader = prepare(cfg, accelerator)
    cfg.freeze()
    # Start training
    train(cfg, train_loader, val_loader, accelerator)


if __name__ == "__main__":
    main()
