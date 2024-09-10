#!/usr/bin/env python3
"""Survey and prepare data from MD trajectories files:

    For trajectories:
        - Check protein topology
        - Number of frames
        - Time between frames

    For each protein, pre-compute following information:    
        - CA contact rate
        - CA pairwise distances (pwd)
        - CA Radius of gyrations (rg)

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

# =============================================================================
# Imports
# =============================================================================

import shutil
import random
import mdtraj
import numpy as np
import pandas as pd
from pathlib import Path
from time import perf_counter, sleep
from argparse import ArgumentParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import Structure

from src.utils.misc import process
from src.utils.protein import protein_io
from src.analysis.mdtraj_tools import get_CA_pairwise_dist, get_radius_of_gyration, filter_CA_pairwise_dist
from src.utils.protein.format import mae_to_pdb
from src.utils.hydra_utils import get_pylogger

logger = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================
from .info import CHAIN_NAME_TO_PROT_NAME, PROTEIN_INFO, PROTEIN_LIST, PWD_NEIGHBOR_EXCLUDE, PWD_BINS, RG_BINS


# =============================================================================
# Functions
# =============================================================================


def survey_traj_info_one_protein(traj_root, prot_code, prot_name, traj_suffix='-c-alpha'):
    """Survey the trajectory information for one protein"""

    traj_root = Path(traj_root)
    prot_info = dict(chain_name=prot_code, prot_name=prot_name)
    traj_dir_list = [d for d in traj_root.glob(f'*{prot_code}*{traj_suffix}') if d.is_dir()]
    prot_info['num_traj'] = len(traj_dir_list)

    seqres = None
    frame_interval_ps = None
    total_frames = 0

    for traj_dir in traj_dir_list:
        traj_name = traj_dir.stem.replace('DESRES-Trajectory_', '')
        time_csv_fpath = traj_dir/traj_name/f"{traj_name}_times.csv"
        pdb_fpath = traj_dir/traj_name/f"{traj_name}.pdb"

        # get seqres
        if not pdb_fpath.exists():
            # convert from MAE file
            mae_fpath = pdb_fpath = traj_dir/traj_name/f"{traj_name}.mae"
            struct = mae_to_pdb(pdb_fpath=pdb_fpath, mae_fpath=mae_fpath)
        else:
            struct = protein_io.load_pdb(pdb_fpath)
        chain = list(struct[0].get_chains())[0] # type:ignore
        seq = ''.join([resi.resname for resi in list(chain)])
        seq = protein_io.seq1(seq)  # 1-letter seq
        if seqres is None:
            seqres = seq
        else:
            assert seqres == seq, f'seqres mismatch for {traj_name}: {seqres} vs {seq}'
        
        # get frame interval in ps
        time_csv = pd.read_csv(time_csv_fpath, names=['start_t', 'fname'])
        frame_interval_ps_ = time_csv['start_t'].min()
        if frame_interval_ps is None:
            frame_interval_ps = frame_interval_ps_
        else:
            assert frame_interval_ps == frame_interval_ps_, \
                f'frame_interval mismatch for {traj_name}: {frame_interval_ps} vs {frame_interval_ps_}'
        
        # load sub-trajectories
        traj_fpath_list = [str(traj_dir/f'{traj_name}/{traj_fpath}') for traj_fpath in time_csv['fname']]
        traj = mdtraj.load(traj_fpath_list, top=str(pdb_fpath))
        coords = traj.xyz
        assert coords.shape[1] == len(seqres)
        total_frames += coords.shape[0]

    prot_info['seqres'] = seqres
    prot_info['seqlen'] = len(seqres)
    prot_info['frame_interval_ps'] = frame_interval_ps
    prot_info['total_frames'] = total_frames
    return pd.Series(prot_info)


def _work_fn_survey_one_protein(arg, traj_root, traj_suffix='-c-alpha'):
    prot_code, prot_name = arg
    return survey_traj_info_one_protein(traj_root, prot_code, prot_name, traj_suffix=traj_suffix)


def survey_metadata_all_proteins(traj_root, output_root, traj_suffix='-c-alpha', n_proc=6):
    arg_list = [(prot_code, prot_name) for prot_name, prot_code, _ in PROTEIN_INFO]
    all_info = process.mp_imap(
        arg_list, _work_fn_survey_one_protein, n_proc=n_proc, 
        traj_root=traj_root, traj_suffix=traj_suffix
    )
    
    all_info = pd.DataFrame(all_info)
    all_info = all_info.set_index('chain_name').reindex(PROTEIN_LIST).reset_index(drop=False)

    output_root = Path(output_root)
    output_root.mkdir(exist_ok=True, parents=True)
    all_info.to_csv(output_root/'metadata.csv', index=False)


def prepare_single_traj_data(traj_dir, output_dir) -> mdtraj.Trajectory:
    """Prepare data for a single trajectory:

        1. Convert MAE topology file to PDB for loading traj in mdtraj
        2. Save all coords
        3. Compute CA pairwise distance
        4. Compute radius of gyration for each atom
    """
    traj_dir = Path(traj_dir)
    traj_name = traj_dir.stem.split('_')[-1]
    output_dir = Path(output_dir)
    assert traj_dir.exists(), f'Traj folder not found: {traj_dir}'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Step 1: prepare pdb topology file and load traj
    mae_fpath = traj_dir/f'{traj_name}/{traj_name}.mae'
    pdb_fpath = traj_dir/f'{traj_name}/{traj_name}.pdb'
    if not pdb_fpath.exists():
        logger.info(f"Converting .mae to .pdb for {mae_fpath}")
        mae_to_pdb(mae_fpath, pdb_fpath)
    if output_dir.joinpath(f"{traj_name}.pdb").exists():
        output_dir.joinpath(f"{traj_name}.pdb").unlink()
    shutil.copy2(pdb_fpath, output_dir)
    # Load sub-trajectories with topology
    traj_time_df = pd.read_csv(traj_dir/f'{traj_name}/{traj_name}_times.csv', names=['time_ps', 'fname'])
    logger.info("Loading trajectories...")
    traj_fpath_list = [str(traj_dir/f'{traj_name}/{traj_fpath}') for traj_fpath in traj_time_df['fname']]
    traj = mdtraj.load(traj_fpath_list, top=str(pdb_fpath))
    n_frames = traj.xyz.shape[0]
    n_atoms = traj.xyz.shape[1]

    # Step 2: save all coords
    coords_fpath = output_dir/'all_coords.npy'
    if not coords_fpath.exists():
        np.save(coords_fpath, traj.xyz)

    # Step 3: get CA pairwise distance
    pdist_fpath = output_dir/'all_pdist.npz'
    if not pdist_fpath.exists():
        pdist, pair_idx = get_CA_pairwise_dist(traj, excluded_neighbors=0) # get all pairs
        assert pdist.shape == (n_frames, n_atoms * (n_atoms - 1) / 2), \
            f'pdist shape mismatch: {pdist.shape} for {n_atoms} atoms {n_frames} frames'
        assert pair_idx.shape == (n_atoms * (n_atoms - 1) / 2, 2), \
            f'pair_idx shape mismatch: {pair_idx.shape} for {n_atoms} atoms {n_frames} frames'
        np.savez_compressed(pdist_fpath, pdist=pdist, pair_idx=pair_idx)

    # Step 4: get radius of gyration (rg) for all atoms
    rg_fpath = output_dir/'all_rg.npy'
    if not rg_fpath.exists():
        rg = get_radius_of_gyration(traj)
        assert rg.shape == (n_frames, n_atoms), \
            f'rg shape mismatch: {rg.shape} for {n_atoms} atoms {n_frames} frames'
        np.save(rg_fpath, rg)

    return traj


def _work_fn_prepare_single_traj_data(arg):
    traj_dir, output_dir = arg
    start_t = perf_counter()
    print(f'Processing {output_dir.name}...')
    prepare_single_traj_data(traj_dir, output_dir)
    end_t = perf_counter()
    print(f"===> {output_dir.name} finished in {end_t - start_t:.1f} s")


def prepare_all_traj_data(traj_root, output_root, suffix='-c-alpha', n_proc=6):
    """Prepare data for all trajectories"""
    traj_root = Path(traj_root)
    traj_dir_list = [d for d in traj_root.glob(f"*{suffix}") if d.is_dir()]
    output_root = Path(output_root)
    output_dir_list = [output_root/(traj_dir.name.split('_')[-1]) for traj_dir in traj_dir_list]

    print(
        f"Processing {len(traj_dir_list)} trajectories:\n  " + 
        '\n  '.join([f'{d} --> {t}' for d, t in zip(traj_dir_list, output_dir_list)])
    )
    sleep(5)
    output_root.mkdir(exist_ok=True, parents=True)
    process.mp_imap(
        iter=list(zip(traj_dir_list, output_dir_list)), 
        func=_work_fn_prepare_single_traj_data, 
        n_proc=n_proc
    )


def _work_fn_prepare_contact_map_ref(row, data_root, cutoff_nm, output_root):
    chain_name = row['chain_name']
    seqlen = row['seqlen']
    output_dir = Path(output_root)/chain_name
    if output_dir.joinpath('contact_map.npy').exists():
        return

    all_pdist = []
    pair_idx = None
    for traj_dir in data_root.glob(chain_name + '*'):
        data = np.load(traj_dir/'all_pdist.npz')
        all_pdist.append(data['pdist'])
        if pair_idx is None:
            pair_idx = data['pair_idx']
    all_pdist = np.concatenate(all_pdist)

    contact_pct = pd.DataFrame(pair_idx, columns=['res_1', 'res_2'])
    contact_pct['contact_rate'] = np.mean(all_pdist < cutoff_nm, axis=0)
    res_list = np.arange(seqlen)
    contact_map = contact_pct.pivot(index='res_1', columns='res_2', values='contact_rate').reindex(index=res_list, columns=res_list).fillna(0)
    contact_map += contact_map.values.T
    contact_map += np.eye(seqlen)
    contact_map = contact_map.values
    np.save(output_dir/'contact_map.npy', contact_map)


def prepare_contact_map_ref(dataset_root, cutoff_A=10., n_proc=12):
    """Prepare the full MD reference contact map for all 12 proteins

    Default cutoff distance is 10 A (str2str)
    """

    dataset_root = Path(dataset_root)
    metadata = pd.read_csv(dataset_root/'metadata.csv')
    data_root = dataset_root/'processed'
    output_dir = dataset_root/'fullmd_ref_value'
    output_dir.mkdir(parents=True, exist_ok=True)

    cutoff_nm = cutoff_A / 10.   

    print("Prepare fullMD reference contact maps...")
    row_list = [row for _, row in metadata.iterrows()]
    process.mp_imap(
        func=_work_fn_prepare_contact_map_ref, iter=row_list, n_proc=n_proc,
        data_root=data_root, cutoff_nm=cutoff_nm, output_dir=output_dir
    )


def prepare_PwD_ref(dataset_root, n_bins=PWD_BINS, n_proc=None):
    """Prepare the full MD (mean) pairwise distance distribution"""

    dataset_root = Path(dataset_root)
    metadata = pd.read_csv(dataset_root/'metadata.csv')
    data_root = dataset_root/'processed'
    output_dir = dataset_root/'fullmd_ref_value'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Prepare full MD reference PwD distributions ...")
    for _, row in metadata.iterrows():
        chain_name = row['chain_name']
        n_frame = row['total_frames']
        ref_fpath = output_dir/f'{chain_name}_pwd_dist.npz'
        if ref_fpath.exists():
            continue
        print(f"Processing {CHAIN_NAME_TO_PROT_NAME[chain_name]}")
        
        # Load pdist
        all_pdist = []
        pair_idx = None
        for traj_dir in data_root.glob(chain_name + '*'):
            data = np.load(traj_dir/'all_pdist.npz')
            pdist, pair_idx_ = filter_CA_pairwise_dist(
                data['pdist'], data['pair_idx'], 
                excluded_neighbors=PWD_NEIGHBOR_EXCLUDE
            )
            if pair_idx is None:
                pair_idx = pair_idx_
            else:
                assert len(pair_idx) == len(pair_idx_)
            all_pdist.append(pdist)
        all_pdist = np.concatenate(all_pdist, axis=0)
        assert len(all_pdist) == n_frame

        # Compute histogram per PwD channel
        results = process.mp_imap(
            func=np.histogram, iter=[all_pdist[:, d] for d in range(all_pdist.shape[1])], n_proc=n_proc,
            bins=n_bins, 
        )
        H = np.concatenate([res[0][None, :] for res in results], axis=0)
        bins = np.concatenate([res[1][None, :] for res in results], axis=0)
        assert H.shape == (all_pdist.shape[1], n_bins)
        assert bins.shape == (all_pdist.shape[1], n_bins + 1)
        np.savez(ref_fpath, H=H, bins=bins, pair_idx=pair_idx)


def prepare_Rg_ref(dataset_root, n_bins=RG_BINS, n_proc=None):
    """Prepare the full MD Jansen-Shannon distance over (mean) pairwise distance"""

    dataset_root = Path(dataset_root)
    metadata = pd.read_csv(dataset_root/'metadata.csv')
    data_root = dataset_root/'processed'
    output_dir = dataset_root/'fullmd_ref_value'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Prepare full MD reference Rg distributions ...")
    for _, row in metadata.iterrows():
        chain_name = row['chain_name']
        seqlen = row['seqlen']
        n_frame = row['total_frames']
        ref_fpath = output_dir/f'{chain_name}_rg_dist.npz'
        if ref_fpath.exists():
            continue

        print(f"Processing {CHAIN_NAME_TO_PROT_NAME[chain_name]}")
        
        # load Rg
        all_rg = []
        for traj_dir in data_root.glob(chain_name + '*'):
            rg = np.load(traj_dir/'all_rg.npy')
            assert rg.shape[1] == seqlen
            all_rg.append(rg)
        all_rg = np.concatenate(all_rg, axis=0)
        assert len(all_rg) == n_frame

        # Compute ref histogram for each channel
        results = process.mp_imap(
            func=np.histogram, iter=[all_rg[:, d] for d in range(all_rg.shape[1])], n_proc=n_proc,
            bins=n_bins, 
        )
        H = np.concatenate([res[0][None, :] for res in results], axis=0)
        bins = np.concatenate([res[1][None, :] for res in results], axis=0)
        assert H.shape == (all_rg.shape[1], n_bins)
        assert bins.shape == (all_rg.shape[1], n_bins + 1)

        np.savez(ref_fpath, H=H, bins=bins)


def extract_example_pdbs(traj_root, dataset_root, n_samples=1000):
    """Extract example pdbs from full atom MD data"""
    traj_root = Path(traj_root)
    dataset_root = Path(dataset_root)
    metadata = pd.read_csv(dataset_root/'metadata.csv')
    output_dir = dataset_root/'examples/pdbs'

    for chain_name in metadata['chain_name']:
        output_root = output_dir/chain_name
        output_root.mkdir(parents=True, exist_ok=True)

        if output_root.joinpath(f"{chain_name}_sample{n_samples - 1}.pdb").exists():
            print(f"[{chain_name}] {n_samples} samples found, skip")
            continue
        traj_dir_list = list(traj_root.glob(f'*{chain_name}*-protein'))
        print(f"Processing {chain_name} with {len(traj_dir_list)} trajectories")
 
        # Prepare topology file and traj_list
        all_traj_list = []
        for traj_dir in traj_dir_list:
            traj_name = traj_dir.name.split('_')[1]
            pdb_top_file = traj_dir/traj_name/f'{traj_name}.pdb'
            if not pdb_top_file.exists():
                mae_top_file = traj_dir/traj_name/f'{traj_name}.mae'
                mae_to_pdb(mae_top_file, pdb_top_file)
            for traj_fpath in traj_dir.rglob('*.dcd'):
                all_traj_list.append((pdb_top_file, traj_fpath))
        print(f"{len(all_traj_list):,} sub-trajectories found")

        # sample structures. to save time, we sample from top max 10 sub-trajectories
        random.shuffle(all_traj_list)
        traj_list = all_traj_list[:min(10, len(all_traj_list))]
        all_traj = [
            mdtraj.load(str(traj_fpath), top=str(pdb_top_file)) for pdb_top_file, traj_fpath in traj_list
        ]
        # include only heavy atoms
        all_traj = [traj.atom_slice(traj.topology.select_atom_indices('heavy')) for traj in all_traj]

        for ix in range(n_samples):
            traj = all_traj[ix % len(all_traj)]
            idx = np.random.choice(np.arange(traj.n_frames), 1)
            traj.slice(idx).save_pdb(str(output_root/f"{chain_name}_sample{ix}.pdb"))
            _fix_pdb_file(output_root/f"{chain_name}_sample{ix}.pdb", chain_name=chain_name)


def _fix_pdb_file(pdb_fpath, chain_name):
    """Fix PDB file for 2WAV, GTT, lambda, NTL9 for openmm processing"""
    if chain_name in ['2WAV', 'GTT', 'lambda', 'NTL9']:
        # rewriting PDB file with correct PDB atom order
        struct = protein_io.load_pdb(pdb_fpath)
        if chain_name == 'NTL9':
            # Remove terminal NH2 and correct element type for O
            struct = protein_io.load_pdb(pdb_fpath)
            chain = struct[0].child_list[0]
            last_res = list(chain.get_residues())[-1]
            for atom in last_res.get_atoms():
                if atom.name == 'NH2':
                    last_res.detach_child(atom.id)
                if atom.name == 'O':
                    atom.element = 'O'
                    atom.name = 'OXT'
        pdb_fpath = Path(pdb_fpath)
        orig_dir = pdb_fpath.parent.joinpath('orig')
        if not orig_dir.exists():
            orig_dir.mkdir()
        shutil.move(pdb_fpath, orig_dir)
        protein_io.write_pdb(struct, fpath=str(pdb_fpath))
    


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = ArgumentParser(prog="Fast-fold MD data preparation")

    parser.add_argument('--traj-root', type=str, required=True)
    parser.add_argument('--output-root', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=6)
    args = parser.parse_args()

    # Step 1: survey and prepare traj data info
    survey_metadata_all_proteins(args.traj_root, args.output_root, n_proc=args.num_workers)
    prepare_all_traj_data(args.traj_root, args.output_root/'processed', n_proc=args.num_workers)


    # Step 2: precompute reference data from full MD
    prepare_contact_map_ref(args.output_root, n_proc=args.num_workers)
    prepare_PwD_ref(args.output_root, n_proc=args.num_workers)
    prepare_Rg_ref(args.output_root, n_proc=args.num_workers)

    # Optional: extract some example PDBs
    # extract_example_pdbs(args.traj_root, args.output_root)