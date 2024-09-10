"""Methods to evaluate conformations against a single reference structure

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

# =============================================================================
# Imports
# =============================================================================
from typing import List, Dict, Union, Tuple, Callable, Optional

import os
import pandas as pd
from pathlib import Path, PosixPath

from . import struct_align, struct_quality, struct_diversity

from src.utils.hydra_utils import get_pylogger
logger = get_pylogger(__name__)


# =============================================================================
# Constants
# =============================================================================

PATH_TYPE = Union[str, PosixPath]


ALIGN_METRICS = ['RMSD', 'TMscore', 'GDT-TS', 'GDT-HA', 'lDDT', 'SC-FAPE']


METRIC_FORMATTER = {
    'avg_RMSD': '{:.2f}',
    'avg_TMscore': '{:.3f}',
    'avg_GDT-TS': '{:.3f}',
    'avg_GDT-HA': '{:.3f}',
    'avg_lDDT': '{:.3f}',
    'avg_SC-FAPE': '{:.3f}',
    'best_RMSD': '{:.2f}',
    'best_TMscore': '{:.3f}',
    'best_GDT-TS': '{:.3f}',
    'best_GDT-HA': '{:.3f}',
    'best_lDDT': '{:.3f}',
    'best_SC-FAPE': '{:.3f}'
}


METRIC_TABLE_NAME = {
    'avg_RMSD': 'RMSD (avg)',
    'avg_TMscore': 'TMscore (avg)',
    'avg_GDT-TS': 'GDT-TS (avg)',
    'avg_GDT-HA': 'GDT-HA (avg)',
    'avg_lDDT': 'lDDT (avg)',
    'avg_SC-FAPE': 'SC-FAPE (avg)',
    'best_RMSD': 'RMSD (best)',
    'best_TMscore': 'TMscore (best)',
    'best_GDT-TS': 'GDT-TS (best)',
    'best_GDT-HA': 'GDT-HA (best)',
    'best_lDDT': 'lDDT (best)',
    'best_SC-FAPE': 'SC-FAPE (best)'
}


# =============================================================================
# Functions
# =============================================================================

def get_best_and_avg(df, best_by='TMscore', cols=None) -> pd.Series:
    """Compute stats (best and avg) for on chain_name case"""
    if cols is None:
        cols = [col for col in ALIGN_METRICS if col in df.columns]
    best = df.sort_values(by=best_by, ascending=False).iloc[0]

    stats = pd.concat([
        df[cols].mean().rename(lambda key: f'avg_{key}'),
        best[cols].rename(lambda key: f'best_{key}')
    ])
    stats['n_sample'] = len(df)
    return stats


def eval_ensemble(
    sample_fpaths: Union[PATH_TYPE, List[PATH_TYPE]], 
    ref_fpath: PATH_TYPE,  
    compute_lddt: bool = True, 
    compute_sc_fape: bool = False, seqres: Optional[str] = None, 
    compute_struct_quality: bool = True, 
    compute_sample_diversity: bool = True, max_pairs=100,
    **align_kwargs
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """Evaluate the alignment against one reference structure, for one protein

    Args:
        sample_fpaths (List[PATH_TYPE]): a list of sampled conformation pdb files or root to a folder of pdb files
        ref_fpath (PATH_TYPE): reference pdb file
        compute_lddt (bool): if compute lddt scores. Default True for single-conf evaluation
        compute_sc_fape (bool): if compute side-chain fape loss. Default False for single-conf evaluation
        seqres (str, optional): reference sequence, required if compute_sc_fape = True
        compute_struct_quality (bool): if also compute structure quality
        compute_sample_diversity (bool): if also sample pairwise alignment among samples as the diversity measure
        max_pairs (int): number of sample pairs to use when estimating sample diversity
        **align_kwargs: other kwargs pass to compute_align_scores

    Returns:
        {
            'metrics': pd.Series of metrics per protein
            'align_scores': pd.DataFrame of alignment to ref struct. 
            'struct_scores': pd.DataFrame of structural quality
            'align_pairwise': pd.DataFrame of pairwise alignment
        }

    """
    assert ref_fpath is not None, "ground truth coords ref_fpath is requried"

    ret_val = {}
    metrics = []

    if isinstance(sample_fpaths, list):
        # List of PDB inputs
        pass
    elif Path(sample_fpaths).is_dir():
        sample_fpaths = [fpath for fpath in sample_fpaths.glob('*.pdb')]
    else:
        # single pdb path
        sample_fpaths = [sample_fpaths]

    # -------------------- align to ref --------------------
    align_scores = pd.DataFrame([
        {
            'fname': Path(sample_fpath).stem,
            **struct_align.compute_align_scores(
                test_struct=sample_fpath, 
                ref_struct=ref_fpath, 
                compute_lddt=compute_lddt, 
                compute_sc_fape=compute_sc_fape, seqres=seqres,
                **align_kwargs
            )
        } for sample_fpath in sample_fpaths
    ])
    metrics.append(get_best_and_avg(align_scores))
    ret_val['align_scores'] = align_scores

    # -------------------- struct quality --------------------
    if compute_struct_quality:
        res = struct_quality.eval_ensemble(sample_fpaths)
        ret_val['struct_scores'] = res['struct_scores']
        metrics.append(res['metrics'])

    # -------------------- diversity --------------------
    if compute_sample_diversity:
        res = struct_diversity.eval_ensemble(
            sample_fpaths, max_samples=max_pairs, 
            compute_lddt=compute_lddt, 
            compute_sc_fape=compute_sc_fape, seqres=seqres,
            n_proc=1,
            **align_kwargs
        )
        ret_val['align_pairwise'] = res['align_pairwise']
        metrics.append(res['metrics'])
    
    ret_val['metrics'] = pd.concat(metrics)

    return ret_val


def report_stats(metrics: pd.DataFrame):
    """Get report table for structural alignment w.r.t. a single reference
    
    It summarizes over the multiple proteins
    """
    report_tab = {}
    for metric_name in METRIC_FORMATTER.keys():
        if metric_name not in metrics.columns:
            continue 
        val_mean = metrics[metric_name].mean()
        val_median = metrics[metric_name].median()

        formatter = METRIC_FORMATTER[metric_name]
        tab_name = METRIC_TABLE_NAME[metric_name]
        report_tab[tab_name] = formatter.format(val_mean) + '/' + formatter.format(val_median)
    return pd.Series(report_tab)

