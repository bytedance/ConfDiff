"""Process RCSB mmcif files to 

1. Extract valid proteins as single chain PDB files
2. Compose a csv file with chain-level metadata info

----------------
[LICENSE]

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

----------------
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). 
All Bytedance's Modifications are Copyright (2024) Bytedance Ltd. and/or its affiliates. 
"""

# =============================================================================
# Imports
# =============================================================================
import os
import math
import random
import argparse
import numpy as np
import pandas as pd
import mdtraj as md
from tqdm import tqdm
from pathlib import Path, PosixPath
from typing import List, Dict, Any, Union

import requests
from Bio.PDB import PDBIO

from src.utils.misc.process import mp_with_timeout
from . import errors, mmcif_parsing
from .utils import prune_chain, chain_to_npy, AtomSelect

# =============================================================================
# Constants
# =============================================================================
_CLUSTER_URL = "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{threshold}.txt"


# =============================================================================
# Functions
# =============================================================================


def _parse_sequence_clustering(path):
    assert os.path.isfile(path), f"File not found: {path}"
    with open(path, 'r') as f:
        cluster_info = f.readlines()
    lookup = {}
    for cluster_id, row in enumerate(cluster_info):
        for instance in row.strip().split():
            if not (len(instance.split('_')) == 2 and len(instance.split('_')[0]) == 4):
                continue
            pdb_id, entity_id = instance.split('_')
            assert pdb_id == pdb_id.upper()
            if lookup.get(pdb_id, None) is not None:
                # the same pdb entity id should not appear in two clusters
                assert entity_id not in lookup[pdb_id], 'duplicate pdb entity ID'
                lookup[pdb_id][entity_id] = cluster_id
            else:
                lookup[pdb_id] = {entity_id: cluster_id}
    return lookup


def get_cluster_lookup(cluster_dir):
    """Download and load cluster lookup files
    
    Cluster files are downloaded from https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{threshold}.txt

    """
    cluster_lookup = {}
    for cluster_threshold in ['30', '50', '70', '90', '100']:
        cluster_fpath = Path(cluster_dir)/f'clusters-by-entity-{cluster_threshold}.txt'
        if not cluster_fpath.exists():
            # Download cluster files
            cluster_fpath.parent.mkdir(parents=True, exist_ok=True)
            url = _CLUSTER_URL.format(threshold=cluster_threshold)
            print(f"Downloading {url} to {cluster_fpath}")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(cluster_fpath, 'w', encoding='utf-8') as f:
                    for chunk in tqdm(r.iter_content(chunk_size=8192)):
                        f.write(chunk.decode('utf-8'))
        print(f"Loading cluster {cluster_threshold} from {cluster_fpath} ...")
        cluster_lookup[cluster_threshold] = _parse_sequence_clustering(
            f'{cluster_dir}/clusters-by-entity-{cluster_threshold}.txt'
        )
    return cluster_lookup


def process_mmcif(
    mmcif_path: Union[str, PosixPath],
    cluster_lookup: Dict[str, Dict],
) -> List[Dict[str, Any]]:

    metadata = [] # return a list of valid chain info
    try:
        # parse mmCIF
        mmcif_object = mmcif_parsing.parse(mmcif_path)
        pdb_id = mmcif_object.pdb_id
        header = mmcif_object.header
        full_structure = mmcif_object.full_structure
        chain_to_seqres = mmcif_object.chain_to_seqres
        entity_to_chains = mmcif_object.entity_to_chains
        struct_mappings = mmcif_object.struct_mappings

        # Parse mmcif header
        mmcif_resolution = header['resolution']
        # if mmcif_resolution == 0.0:
        #     raise errors.ResolutionError(f'Invalid resolution {mmcif_resolution}')
        if mmcif_resolution > args.max_resolution:
            raise errors.ResolutionError(f'Poor resolution: {mmcif_resolution}')

        # assure all models have the same chains
        if len(full_structure.child_list) > 1:
            model0_chains = set(full_structure.child_list[0].child_dict.keys())
            for model_idx in range(1, len(full_structure.child_list)):
                modelx_chains = set(full_structure.child_list[model_idx].child_dict.keys())
                assert model0_chains == modelx_chains, 'Different chains across models.'
        
        for model_id, model_map in struct_mappings.items():
            assert model_id in full_structure.child_dict, 'Model id mismatch.'
            for chain_id, chain_map in model_map.items():
                seqres = chain_to_seqres[chain_id]
                num_X = seqres.count('X')
                if not (args.min_len <= (len(seqres) - num_X) and \
                        float(num_X) / len(seqres) < 0.5 and \
                        len(seqres) <= args.max_len):
                    # skip under-/oversized and low-quality chains
                    continue
                assert len(seqres) == len(chain_map), 'len(seqres) != len(chain_map)'
                chain = full_structure.child_dict[model_id].child_dict[chain_id] # biopython model ids are zero-based integers
                chain_name = f'{pdb_id}_{model_id}_{chain_id}' # unique chain identifier

                # Biopython chain -> numpy atom coords
                chain_info = chain_to_npy(
                    seqres=seqres,
                    chain=chain,
                    chain_map=chain_map
                )
                atom_coords = chain_info['atom_coords']
                atom_mask = np.all(~np.isnan(atom_coords), axis=-1)
                num_valid_frames = np.all(atom_mask[:, [0, 1, 2, 4]], axis=-1).sum()
                valid_frame_ratio = num_valid_frames / len(seqres)
                if (valid_frame_ratio < 0.5) or \
                   (num_valid_frames < args.min_len): 
                    continue # too many missing residues
                
                if args.output_pdb_dir is not None:
                    pdb_subdir = os.path.join(args.output_pdb_dir, pdb_id[1:3], pdb_id)
                    os.makedirs(pdb_subdir, exist_ok=True)
                    pdb_path = os.path.join(pdb_subdir, f'{chain_name}.pdb')
                    assert not os.path.exists(pdb_path), f'{pdb_path} already exists.'
                    try:
                        # Biopython chain -> PDB file
                        chain_pruned = prune_chain(
                            chain=chain,
                            chain_map=chain_map
                        )
                        if len(list(chain_pruned)) < args.min_len: continue
                        # save as PDB file
                        pdbio = PDBIO()
                        pdbio.set_structure(chain_pruned)
                        pdbio.save(pdb_path, select=AtomSelect())
                        # SS calculation
                        traj = md.load(pdb_path)
                        pdb_ss = md.compute_dssp(traj, simplified=True)
                        coil_ratio = np.sum(pdb_ss == 'C') / len(seqres)
                        helix_ratio = np.sum(pdb_ss == 'H') / len(seqres)
                        strand_ratio = np.sum(pdb_ss == 'E') / len(seqres)
                    except:
                        # skip chains that could not be saved by PDBIO
                        Path(pdb_path).unlink(missing_ok=True)
                        continue

                # sequence-based clustering
                chain_entity_id = [entity_id for entity_id, chains in entity_to_chains.items() if chain_id in chains]
                assert len(chain_entity_id) == 1, 'len(chain_entity_id) != 1'
                chain_entity_id = chain_entity_id[0] # str
                try:
                    cluster_30_id = cluster_lookup['30'][pdb_id][chain_entity_id]
                except:
                    cluster_30_id = -1
                try:
                    cluster_50_id = cluster_lookup['50'][pdb_id][chain_entity_id]
                except:
                    cluster_50_id = -1
                try:
                    cluster_70_id = cluster_lookup['70'][pdb_id][chain_entity_id]
                except:
                    cluster_70_id = -1
                try:
                    cluster_90_id = cluster_lookup['90'][pdb_id][chain_entity_id]
                except:
                    cluster_90_id = -1
                try:
                    cluster_100_id = cluster_lookup['100'][pdb_id][chain_entity_id]
                except:
                    cluster_100_id = -1
                
                info = {
                    'chain_name': chain_name, # unique chain identifier
                    'pdb_id': pdb_id,
                    'model_id': model_id,
                    'chain_id': chain_id,
                    'structure_method': header['structure_method'],
                    'resolution': header['resolution'],
                    'release_date': header['release_date'],
                    'seqres': seqres,
                    'seqlen': len(seqres),
                    'valid_frame_ratio': valid_frame_ratio,
                    'coil_ratio': coil_ratio,
                    'helix_ratio': helix_ratio,
                    'strand_ratio': strand_ratio,
                    'cluster_30_id': cluster_30_id,
                    'cluster_50_id': cluster_50_id,
                    'cluster_70_id': cluster_70_id,
                    'cluster_90_id': cluster_90_id,
                    'cluster_100_id': cluster_100_id,
                }
                metadata.append(info)
        
        # return metadata
        return metadata
    
    # handle errors
    except errors.HeaderError as e:
        print(f'HeaderError at {mmcif_path}: {e}', flush=True)
    except errors.NoProteinError as e:
        pass
        #print(f"NoProteinError at {mmcif_path}: {e}", flush=True)
    except errors.ResolutionError as e:
        print(f"ResolutionError at {mmcif_path}: {e}", flush=True)
    except errors.ResidueError as e:
        print(f"ResidueError at {mmcif_path}: {e}", flush=True)
    except AssertionError as e:
        print(f"Assertion failed at {mmcif_path}: {e}", flush=True)
    except Exception as e:
        print(f"Exception occured at {mmcif_path}: {e}", flush=True)
    return None


# =============================================================================
# Main
# =============================================================================


def main(args):
    
    all_mmcif_paths = list(Path(args.mmcif_dir).rglob('*.cif.gz'))

    random.shuffle(all_mmcif_paths) # balance worker load
    if args.debug:
        num_samples = math.ceil(0.001*len(all_mmcif_paths))
        all_mmcif_paths = all_mmcif_paths[:num_samples]
    print(f'Total number of mmcif files: {len(all_mmcif_paths)}')

    cluster_lookup = get_cluster_lookup(args.cluster_dir)

    metadata = []
    if args.num_workers > 1:
        results = mp_with_timeout(
            iter=all_mmcif_paths, func=process_mmcif, n_proc=args.num_workers,
            timeout=args.timeout, chunksize=1, print_every_iter=10000, error_report_fn=lambda x: x,
            # args
            cluster_lookup=cluster_lookup
        )
        for res in results:
            if isinstance(res, list):
                metadata += res
    else:
        for path in tqdm(all_mmcif_paths):
            res = process_mmcif(path)
            if isinstance(res, list):
                metadata += res
    
    # save metadata to csv
    df = pd.DataFrame(metadata)
    df.to_csv(args.output_csv_path, index=False)
    print(df)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description='Process mmCIF.'
    )
    parser.add_argument(
        '--mmcif_dir',
        help='Path to directory with mmcif files.',
        type=str,
        required=True)
    parser.add_argument(
        '--cluster_dir',
        help='Path to directory with sequence-based clustering txt files.',
        type=str,
        required=True)
    parser.add_argument(
        '--output_csv_path',
        help='Path to output metadata csv file.',
        type=str,
        required=True)
    parser.add_argument(
        '--output_pdb_dir',
        help='Path to output PDB file dir.',
        type=str,
        default=None)
    parser.add_argument(
        '--min_len',
        help='Min length of protein chain.',
        type=int,
        default=20)
    parser.add_argument(
        '--max_len',
        help='Max length of protein chain.',
        type=int,
        default=512)
    parser.add_argument(
        '--max_resolution',
        help='Max resolution.',
        type=float,
        default=5.)
    parser.add_argument(
        '--num_workers',
        help='Number of workers.',
        type=int,
        default=1)
    parser.add_argument(
        '--timeout',
        help='Patience (sec) before timeout.',
        type=int,
        default=300)
    parser.add_argument(
        '--debug',
        help='Debug mode. Use 0.001 of all data',
        default=False,
        action="store_true")
    args = parser.parse_args()

    main(args)
