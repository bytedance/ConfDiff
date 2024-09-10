"""Methods for TICA analysis of fast folding proteins

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
from typing import List, Optional, Tuple
from time import perf_counter
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from deeptime.decomposition import TICA
from src.analysis.md_dist import compute_hist_counts_2d
from src.utils.misc import process
from src.utils.misc.misc import save_pickle, load_pickle

# =============================================================================
# Constants
# =============================================================================
from datasets.fastfold.info import DATASET_DIR, TICA_LAGTIME, TICA_BINS


DEFAULT_LAGTIME_CANDIDATES = [
    20,
    100, 160, 250, 400, 630, 
    1000, 1600, 2500, 4000, 6300, 
    10000, 16000, 25000, 40000, 63000, 
    100000,
] # max 200 ps * 100000 = 60 us


# =============================================================================
# Functions
# =============================================================================


def _get_lagtime(lagtime, chain_name):
    if lagtime == 'best':
        lagtime = TICA_LAGTIME[chain_name]
    elif lagtime == 'str2str':
        lagtime = 20
    else:
        assert isinstance(lagtime, int)
    return lagtime


def get_tica_model(chain_name, lagtime='best', dataset_root=DATASET_DIR):
    """Fetch TICA model and pre-computed reference distribution"""
    dataset_root = Path(dataset_root)
    lagtime = _get_lagtime(lagtime, chain_name)
    model_fpath = dataset_root/f'tica/{chain_name}/{chain_name}_lagtime{lagtime}_model.pkl'
    model = load_pickle(model_fpath)
    ref_dist_fpath = dataset_root/f'tica/{chain_name}/{chain_name}_lagtime{lagtime}_dist_2d.npz'
    dist_2d = dict(np.load(ref_dist_fpath))
    ref_dist_fpath = dataset_root/f'tica/{chain_name}/{chain_name}_lagtime{lagtime}_dist_1d.npz'
    dist_1d = dict(np.load(ref_dist_fpath))
    return dict(model=model, dist_2d=dist_2d, dist_1d=dist_1d)


def get_fullmd_tica_proj(chain_name, lagtime='best', dataset_root=DATASET_DIR):
    """Fetch Full MD TICA projections"""
    lagtime = _get_lagtime(lagtime, chain_name)
    proj_fpath = Path(dataset_root).joinpath('tica', chain_name, f'{chain_name}_lagtime{lagtime}_proj.npy')
    return np.load(proj_fpath)
 

def fit_tica(
    data_root, chain_name, num_traj, lagtime, 
    dim=None, suffix='-c-alpha', 
    **tica_kwargs
) -> TICA:
    """Fit TICA model using deeptime.decomposition.TICA"""

    data_root = Path(data_root)
    
    tica = TICA(lagtime=lagtime, dim=dim, scaling='kinetic_map', **tica_kwargs)

    all_pdist_data = []

    for traj_idx in range(num_traj):
        # load data
        data_dir = data_root/f'{chain_name}-{traj_idx}{suffix}'
        data = np.load(data_dir/'all_pdist.npz')
        pdist = data['pdist']
        pair_idx = data['pair_idx']
        n_samples = pdist.shape[0] - lagtime
        all_pdist_data.append(pdist)
        if n_samples <= 10:
            continue
        print(f"Fitting traj {traj_idx} with {pdist.shape[0]:,} frames ({n_samples:,} lagged data) ...")
        tica = tica.partial_fit(pdist)
        del pair_idx
    
    all_proj_data = []
    for pdist in all_pdist_data:
        all_proj_data.append(tica.transform(pdist)[:, :2])
    all_proj_data = np.concatenate(all_proj_data)

    return tica, all_proj_data


def run_lagtime_scan(
    metadata, data_root, output_root, 
    lagtime_list=DEFAULT_LAGTIME_CANDIDATES, hist_bins=TICA_BINS,
    **kwargs
):
    """Scan different lag times (in number of frames) for TICA
    """
    for _, row in metadata.iterrows():
        prot_name = row['prot_name']
        chain_name = row['chain_name']
        num_traj = row['num_traj']
        output_dir = output_root/chain_name
        output_dir.mkdir(parents=True, exist_ok=True)

        start_t = perf_counter()
        print(
            '\n' +
            '#' * 70 + '\n' + 
            "#" + f'{prot_name} ({chain_name}), {num_traj} trajectories'.center(68) + '#\n' +
            '#' * 70 + '\n' +
            '\n'
        )
        
        for lagtime in lagtime_list:
            print(f'\n##### lagtime = {lagtime} frames #####\n')
            model_fpath = output_dir/f'{chain_name}_lagtime{lagtime}_model.pkl'
            if not model_fpath.exists():
                tica, all_proj_data = fit_tica(data_root, chain_name, num_traj, lagtime, **kwargs)
                save_pickle(tica, output_dir/f'{chain_name}_lagtime{lagtime}_model.pkl')
                np.save(output_dir/f'{chain_name}_lagtime{lagtime}_proj.npy', all_proj_data)
            
                # compute dist
                H, xs, ys, xedges, yedges = compute_hist_counts_2d(all_proj_data[:, :2], bins=hist_bins)
                np.savez(output_dir/f'{chain_name}_lagtime{lagtime}_dist_2d.npz', H=H, xedges=xedges, yedges=yedges)

                H1, bins1 = np.histogram(all_proj_data[:, 0], bins=hist_bins)
                H2, bins2 = np.histogram(all_proj_data[:, 1], bins=hist_bins)
                np.savez(
                    output_dir/f"{chain_name}_lagtime{lagtime}_dist_1d.npz",
                    H=np.concatenate([H1[None, ...], H2[None, ...]], axis=0),
                    bins=np.concatenate([bins1[None, ...], bins2[None, ...]], axis=0)
                )

        end_t = perf_counter()
        print(f"Done in {end_t - start_t:.1f} sec\n")


def run_best_tica(
    metadata, data_root, output_root, 
    hist_bins=TICA_BINS,
    **kwargs
):
    """Run tica fitting for the best lag time (in number of frames) for TICA
    """
    for _, row in metadata.iterrows():
        prot_name = row['prot_name']
        chain_name = row['chain_name']
        num_traj = row['num_traj']
        output_dir = output_root/chain_name
        output_dir.mkdir(parents=True, exist_ok=True)
        lagtime = TICA_LAGTIME[chain_name]

        start_t = perf_counter()
        print(
            '\n' +
            '#' * 70 + '\n' + 
            "#" + f'{prot_name} ({chain_name}), {num_traj} trajectories, lagtime = {lagtime}'.center(68) + '#\n' +
            '#' * 70 + '\n' +
            '\n'
        )

        model_fpath = output_dir/f'{chain_name}_lagtime{lagtime}_model.pkl'
        if not model_fpath.exists():
            tica, all_proj_data = fit_tica(data_root, chain_name, num_traj, lagtime, **kwargs)
            save_pickle(tica, output_dir/f'{chain_name}_lagtime{lagtime}_model.pkl')
            np.save(output_dir/f'{chain_name}_lagtime{lagtime}_proj.npy', all_proj_data)
        
            # compute dist
            H, xs, ys, xedges, yedges = compute_hist_counts_2d(all_proj_data[:, :2], bins=hist_bins)
            np.savez(output_dir/f'{chain_name}_lagtime{lagtime}_dist_2d.npz', H=H, xedges=xedges, yedges=yedges)

            H1, bins1 = np.histogram(all_proj_data[:, 0], bins=hist_bins)
            H2, bins2 = np.histogram(all_proj_data[:, 1], bins=hist_bins)
            np.savez(
                output_dir/f"{chain_name}_lagtime{lagtime}_dist_1d.npz",
                H=np.concatenate([H1[None, ...], H2[None, ...]], axis=0),
                bins=np.concatenate([bins1[None, ...], bins2[None, ...]], axis=0)
            )
        else:
            print(f"Model {model_fpath} found. Skip")
        
        end_t = perf_counter()
        print(f"Done in {end_t - start_t:.1f} sec\n")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':

    parser = ArgumentParser(prog='Fast-folding model selection')
    parser.add_argument('--dataset-root', required=True)
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    metadata = pd.read_csv(dataset_root/'metadata.csv')

    # Run tica fitting with preset lagtime
    run_best_tica(
        metadata, 
        data_root=dataset_root/'processed', 
        output_root=dataset_root/'tica'
    )

    # # Screening lagtime
    # run_lagtime_scan(
    #     metadata, 
    #     data_root=dataset_root/'processed', 
    #     output_root=dataset_root/'tica'
    # ) 

