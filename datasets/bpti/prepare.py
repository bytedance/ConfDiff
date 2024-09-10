#!/usr/bin/env python3
"""Prepare precomputed data from MD trajectories. Includes

    - CA coordinates
    - CA pairwise distance
    - CA radius of gyration (relative the center of mass)

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
import mdtraj
import random
import numpy as np
import pandas as pd
from pathlib import Path
from time import perf_counter, sleep
from argparse import ArgumentParser

from src.utils.misc import process
from src.analysis.md_dist import filter_CA_pairwise_dist
from ..fastfold.prepare import prepare_single_traj_data, survey_traj_info_one_protein, mae_to_pdb

from src.utils.hydra_utils import get_pylogger
logger = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================
from .info import PWD_BINS, RG_BINS, SEQLEN, PWD_NEIGHBOR_EXCLUDE


# =============================================================================
# Functions
# =============================================================================


def prepare_bpti_data(traj_root, output_root, suffix='-c-alpha'):
    """Prepare data for BPTI"""
    traj_root = Path(traj_root)
    traj_dir_list = [d for d in traj_root.glob(f"*{suffix}") if d.is_dir()]
    assert len(traj_dir_list) == 1
    traj_dir = traj_dir_list[0]
    
    output_root = Path(output_root)
    output_dir = output_root/'processed'

    print(
        f"Processing BPTI trajectory:\n  " + 
        f'  {traj_dir} --> {output_dir}' 
    )
    sleep(5)


    start_t = perf_counter()
    print(f'Processing {output_dir.name}...')
    prepare_single_traj_data(traj_dir, output_dir)
    end_t = perf_counter()
    print(f"===> {output_dir.name} finished in {end_t - start_t:.1f} s")


def make_bpti_csv(traj_root, output_root, traj_suffix='-c-alpha'):
    bpti_info = [survey_traj_info_one_protein(traj_root=traj_root, prot_code='bpti', prot_name='BPTI', traj_suffix=traj_suffix)]
    bpti_info = pd.DataFrame(bpti_info)

    output_root = Path(output_root)
    output_root.mkdir(exist_ok=True, parents=True)
    bpti_info.to_csv(output_root/'metadata.csv', index=False)


def prepare_contact_map_ref(dataset_root, cutoff_A=10.):
    """Prepare the full MD reference contact map for all 12 proteins"""

    dataset_root = Path(dataset_root)
    data_root = dataset_root/'processed'
    output_dir = dataset_root/'fullmd_ref_value'
    output_dir.mkdir(parents=True, exist_ok=True)
    seqlen = SEQLEN

    cutoff_nm = cutoff_A / 10.   

    print("Prepare fullMD reference contact maps...")
    
    if output_dir.joinpath('contact_map.npy').exists():
        return

    data = np.load(data_root/'all_pdist.npz')
    pdist = data['pdist']
    pair_idx = data['pair_idx']

    contact_pct = pd.DataFrame(pair_idx, columns=['res_1', 'res_2'])
    contact_pct['contact_rate'] = np.mean(pdist < cutoff_nm, axis=0)
    res_list = np.arange(seqlen)
    contact_map = contact_pct.pivot(index='res_1', columns='res_2', values='contact_rate').reindex(index=res_list, columns=res_list).fillna(0)
    contact_map += contact_map.values.T
    contact_map += np.eye(seqlen)
    contact_map = contact_map.values
    np.save(output_dir/'contact_map.npy', contact_map)
    

def prepare_PwD_ref(dataset_root, n_bins=PWD_BINS, n_proc=None):
    """Prepare the full MD (mean) pairwise distance distribution"""

    dataset_root = Path(dataset_root)
    data_root = dataset_root/'processed'
    output_dir = dataset_root/'fullmd_ref_value'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Prepare full MD reference PwD distributions ...")

    ref_fpath = output_dir/'pwd_dist.npz'
    if ref_fpath.exists():
        return
    
    data = np.load(data_root/'all_pdist.npz')
    pdist, pair_idx = filter_CA_pairwise_dist(
        data['pdist'], data['pair_idx'], 
        excluded_neighbors=PWD_NEIGHBOR_EXCLUDE
    )

    # Compute histogram per PwD channel
    results = process.mp_imap(
        func=np.histogram, iter=[pdist[:, d] for d in range(pdist.shape[1])], n_proc=n_proc,
        bins=n_bins, 
    )
    H = np.concatenate([res[0][None, :] for res in results], axis=0)
    bins = np.concatenate([res[1][None, :] for res in results], axis=0)
    assert H.shape == (pdist.shape[1], n_bins)
    assert bins.shape == (pdist.shape[1], n_bins + 1)
    np.savez(ref_fpath, H=H, bins=bins, pair_idx=pair_idx)


def prepare_Rg_ref(dataset_root, n_bins=RG_BINS, n_proc=None):
    """Prepare the full MD Jansen-Shannon distance over (mean) pairwise distance"""

    dataset_root = Path(dataset_root)
    data_root = dataset_root/'processed'
    output_dir = dataset_root/'fullmd_ref_value'
    output_dir.mkdir(parents=True, exist_ok=True)
    seqlen = SEQLEN

    print("Prepare full MD reference Rg distributions ...")
    ref_fpath = output_dir/'rg_dist.npz'
    if ref_fpath.exists():
        return
    
    # load Rg
    rg = np.load(data_root/'all_rg.npy')
    assert rg.shape[1] == seqlen

    # Compute ref histogram for each channel
    results = process.mp_imap(
        func=np.histogram, iter=[rg[:, d] for d in range(rg.shape[1])], n_proc=n_proc,
        bins=n_bins, 
    )
    H = np.concatenate([res[0][None, :] for res in results], axis=0)
    bins = np.concatenate([res[1][None, :] for res in results], axis=0)
    assert H.shape == (rg.shape[1], n_bins)
    assert bins.shape == (rg.shape[1], n_bins + 1)

    np.savez(ref_fpath, H=H, bins=bins)


def extract_example_pdbs(traj_root, dataset_root, n_samples=1000):
    """Extract example pdbs from full atom MD data"""
    traj_root = Path(traj_root)
    dataset_root = Path(dataset_root)
    output_dir = dataset_root/'example_pdbs'

    chain_name = 'bpti'
    output_root = output_dir/chain_name
    output_root.mkdir(parents=True, exist_ok=True)

    if output_root.joinpath(f"{chain_name}_sample{n_samples - 1}.pdb").exists():
        print(f"[{chain_name}] {n_samples} samples found, skip")
        return 

    traj_dir_list = list(traj_root.glob(f'*{chain_name}*-protein*'))
    print(f"Processing {chain_name} with {len(traj_dir_list)} trajectories")
    # Prepare topology file 
    for traj_dir in traj_dir_list:
        traj_name = traj_dir.name.split('_')[1]
        mae_top_file = traj_dir/traj_name/f'{traj_name}.mae'
        pdb_top_file = traj_dir/traj_name/f'{traj_name}.pdb'
        if not pdb_top_file.exists():
            mae_to_pdb(mae_top_file, pdb_top_file)
    all_traj_list = [traj_fpath for traj_dir in traj_dir_list for traj_fpath in traj_dir.rglob('*.dcd')]

    # sample structures
    random.shuffle(all_traj_list)
    traj_list = all_traj_list[:min(10, len(all_traj_list))]
    traj = mdtraj.load([str(traj_fpath) for traj_fpath in traj_list], top=str(pdb_top_file))
    non_H_atoms = traj.topology.select_atom_indices('heavy')
    idx_selc = np.random.choice(np.arange(traj.n_frames), n_samples)
    traj = traj.atom_slice(non_H_atoms)
    for ix, idx in enumerate(idx_selc):
        traj.slice(idx).save_pdb(str(output_root/f"{chain_name}_sample{ix}.pdb"))

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = ArgumentParser(prog="BPTI MD data preparation")

    parser.add_argument('--traj-root', type=str, required=True)
    parser.add_argument('--output-root', type=str, required=True)
    args = parser.parse_args()

    # Get traj info
    prepare_bpti_data(args.traj_root, args.output_root)
    make_bpti_csv(args.traj_root, args.output_root)

    # Prepare full MD reference data
    prepare_contact_map_ref(args.output_root)
    prepare_PwD_ref(args.output_root)
    prepare_Rg_ref(args.output_root)

    # extract_example_pdbs(args.traj_root, args.output_root)

