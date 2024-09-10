"""Lagtime selection for BPTI's proper time lag

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

# =============================================================================
# Imports
# =============================================================================
from typing import List, Optional, Tuple
from time import perf_counter
from pathlib import Path, PosixPath
from argparse import ArgumentParser
from tqdm  import tqdm

import pickle
import numpy as np
import pandas as pd

from deeptime.decomposition import TICA

from src.analysis.md_dist import compute_hist_counts_2d
from src.utils.misc import process
from src.utils.misc.misc import save_pickle, load_pickle

# =============================================================================
# Constants
# =============================================================================
from .info import TICA_BINS, TICA_LAGTIME, DATASET_DIR

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


def _get_lagtime(lagtime):
    if lagtime == 'best':
        lagtime = TICA_LAGTIME
    elif lagtime == 'str2str':
        lagtime = 20
    else:
        assert isinstance(lagtime, int)
    return lagtime


def get_tica_model(lagtime='best', dataset_root=DATASET_DIR):
    """Fetch pretrained TICA model"""
    dataset_root = Path(dataset_root)
    lagtime = _get_lagtime(lagtime)
    model_fpath = dataset_root/f'tica/bpti_lagtime{lagtime}_model.pkl'
    model = load_pickle(model_fpath)
    ref_dist_fpath = dataset_root/f'tica/bpti_lagtime{lagtime}_dist_2d.npz'
    dist_2d = dict(np.load(ref_dist_fpath))
    ref_dist_fpath = dataset_root/f'tica/bpti_lagtime{lagtime}_dist_1d.npz'
    dist_1d = dict(np.load(ref_dist_fpath))
    return dict(model=model, dist_2d=dist_2d, dist_1d=dist_1d)


def get_fullmd_tica_proj(lagtime='best', dataset_root=DATASET_DIR):
    """Fetch Full MD TICA projections"""
    lagtime = _get_lagtime(lagtime)
    proj_fpath = f'{dataset_root}/tica/bpti_lagtime{lagtime}_proj.npy'
    return np.load(proj_fpath)


def fit_tica_bpti(
    data_root, lagtime, 
    dim=None, suffix='-c-alpha', 
    **tica_kwargs
) -> Tuple[TICA, np.ndarray]:
    """Run K-Fold cross validation on TICA model
    If multiple trajectories are found, K folds of data are generated within each trajectory and update the model through partial_fit

    Returns:
        List of TICA model instance
        DataFrame with validation auto-correlation score
    """

    data_root = Path(data_root)
    
    tica = TICA(lagtime=lagtime, dim=dim, scaling='kinetic_map', **tica_kwargs)

    # load data
    data_dir = data_root/'processed'
    data = np.load(data_dir/'all_pdist.npz')
    pdist = data['pdist']
    pair_idx = data['pair_idx']
    n_samples = pdist.shape[0] - lagtime
    if n_samples <= 10:
        print("Not sufficient data")
        return None, None

    print(f"Fitting traj 0 with {pdist.shape[0]:,} frames ({n_samples:,} lagged data) ...")
    tica = tica.partial_fit(pdist)
    del pair_idx 

    proj_data = tica.transform(pdist)[:, :2]

    return tica, proj_data    


def _work_fn_fit_tica(lagtime, data_root, output_dir, hist_bins=TICA_BINS, **kwargs):
    print(f'\n##### lagtime = {lagtime} frames #####\n')
    model_fpath = output_dir/f'bpti_lagtime{lagtime}_model.pkl'
    if not model_fpath.exists():
        tica, all_proj_data = fit_tica_bpti(data_root=data_root, lagtime=lagtime, **kwargs)
        save_pickle(tica, output_dir/f'bpti_lagtime{lagtime}_model.pkl')
        np.save(output_dir/f'bpti_lagtime{lagtime}_proj.npy', all_proj_data)

        # compute dist
        H, xs, ys, xedges, yedges = compute_hist_counts_2d(all_proj_data[:, :2], bins=hist_bins)
        np.savez(output_dir/f'bpti_lagtime{lagtime}_dist_2d.npz', H=H, xedges=xedges, yedges=yedges)

        H1, bins1 = np.histogram(all_proj_data[:, 0], bins=hist_bins)
        H2, bins2 = np.histogram(all_proj_data[:, 1], bins=hist_bins)
        np.savez(
            output_dir/f"bpti_lagtime{lagtime}_dist_1d.npz",
            H=np.concatenate([H1[None, ...], H2[None, ...]], axis=0),
            bins=np.concatenate([bins1[None, ...], bins2[None, ...]], axis=0)
        )


def run_lagtime_scan(data_root, output_dir, lagtime_list=DEFAULT_LAGTIME_CANDIDATES, **kwargs):
    """Scan different lag times (in number of frames) for TICA through K-Fold cross validation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    process.mp_imap_unordered(
        func=_work_fn_fit_tica, iter=lagtime_list, chunksize=1, n_proc=1,
        data_root=data_root, output_dir=output_dir, **kwargs
    )

def run_best_tica(
    data_root, output_root, 
    **kwargs
):
    """Run tica fitting for the best lag time (in number of frames) for TICA
    """
    output_root.mkdir(parents=True, exist_ok=True)
    lagtime = TICA_LAGTIME

    start_t = perf_counter()
    model_fpath = output_root/f'bpti_lagtime{lagtime}_model.pkl'
    if not model_fpath.exists():
        _work_fn_fit_tica(lagtime=lagtime,  data_root=data_root, output_dir=output_root)
    else:
        print(f"Model {model_fpath} found. Skip")
    end_t = perf_counter()
    print(f"Done in {end_t - start_t:.1f} sec\n")

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':

    parser = ArgumentParser(prog='BPTI TICA fitting')
    parser.add_argument('--dataset-root', required=True)
    args = parser.parse_args()

    args.dataset_root = Path(args.dataset_root)

    # Run TICA with pre-selected lagtime
    run_best_tica(args.dataset_root, output_root=args.dataset_root/'tica')
    
    # # Or run a full lagtime scan
    # run_lagtime_scan(args.dataset_root, output_dir=args.dataset_root/'tica')

