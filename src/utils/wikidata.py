#          Linking Surface Facts to Large-Scale Knowledge Graphs
# 
#   file: wikidata.py
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
import string
from abc import ABC, abstractmethod
from os.path import join as pjoin

import indexed_gzip as igzip
from transformers import RobertaTokenizerFast


class Wikidata(ABC):
    @abstractmethod
    def get_data(self, data):
        pass

    @abstractmethod
    def get_label(self, data):
        pass

    @abstractmethod
    def get_description(self, data):
        pass

    @abstractmethod
    def get_aliases(self, data):
        pass


class WikidataProcessed(Wikidata):
    def __init__(self, wikidata_path: str, **kwargs):
        self.wikidata = json.load(
            open(pjoin(wikidata_path, "entity_rel_data_w_synthie.json"), "r")
        )

    def get_data(self, object_id: str):
        return self.wikidata[object_id]

    def get_label(self, data):
        return data["label"]

    def get_description(self, data):
        return data["description"]

    def get_aliases(self, data):
        return data["aliases"]


class WikidataZip(Wikidata):
    def __init__(self, wikidata_path: str, **kwargs):
        # Load stuff
        self.wikidata = json.load(open(pjoin(wikidata_path, "wikidata_index.json")))
        self.zip_file = igzip.IndexedGzipFile(
            pjoin(wikidata_path, "latest-all.json.gz"),
            index_file=pjoin(wikidata_path, "wikidata_seek_index.gzidx"),
        )
        self.punc = set(string.punctuation)
        # Used just for alias & description filtering (abnormaly long sequences)
        self.tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        # Hard coded, qualitatively inspected that everything longer is random
        # characters or just bad
        self.threshold = kwargs.pop("threshold", 30)

    def get_triplet_aliases(self, fact_ids):
        subj_id, obj_id = fact_ids[0], fact_ids[2]
        # Get data
        subj_data = self.get_data(subj_id)
        obj_data = self.get_data(obj_id)
        # Get aliases
        subj_aliases = self.get_aliases(subj_data)
        obj_aliases = self.get_aliases(obj_data)

        return subj_aliases, obj_aliases

    def get_description(self, data):
        # data is obtained from get_data, given an entity or relation id
        if not data:
            return None
        if "en" in data["descriptions"]:
            description = data["descriptions"]["en"]["value"]
        else:
            description = "None"
        # Check if abnormal tokens
        tokenized = self.tokenizer.encode(description)
        if len(tokenized) > self.threshold:
            tokenized = tokenized[1 : self.threshold]
        # Decode back to string
        description = self.tokenizer.decode(tokenized)

        return description

    def get_label(self, data):
        # data is obtained from get_data, given an entity or relation id
        if not data:
            return None
        if "en" in data["labels"]:
            label = data["labels"]["en"]["value"]
        else:
            label = None

        return label

    def get_aliases(self, data):
        # data is obtained from get_data, given an entity or relation id
        if not data:
            return []
        if "en" not in data["aliases"]:
            return []
        return [
            a["value"]
            for a in data["aliases"]["en"]
            if all(x.isalpha() or x.isspace() or x in self.punc for x in a["value"])
            and len(self.tokenizer.encode(a["value"])) < self.threshold
        ]

    def get_data(self, object_id: str):
        try:
            offset, length = self.wikidata[object_id]
            # Seek to the location
            self.zip_file.seek(offset)
            # Obtain the data chunk
            object_bytes = self.zip_file.read(length)
            # Load the data from the byte array
            data = json.loads(object_bytes[:-2])
        except KeyError:
            data = None

        return data


wikidata_factory = {"zip": WikidataZip, "processed": WikidataProcessed}
