"""Methods to evaluate conformations against two reference structures

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path, PosixPath
from tqdm import tqdm

from . import struct_align, struct_quality, struct_diversity

from src.utils.hydra_utils import get_pylogger

logger = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================
PATH_TYPE = Union[str, PosixPath]

METRIC_FORMATTER = {
    # align_to_ref
    'TMscore_ref1_best': '{:.3f}',
    'TMscore_ref2_best': '{:.3f}',
    'RMSD_ref1_best': '{:.2f}',
    'RMSD_ref2_best': '{:.2f}',
    'TMens': '{:.3f}',
    'TMmin': '{:.3f}',
}


METRIC_TABLE_NAME = {
    # align_to_ref
    'TMscore_ref1_best': 'Best ref1 TMscore',
    'TMscore_ref2_best': 'Best ref2 TMscore',
    'RMSD_ref1_best': 'Best ref1 RMSD',
    'RMSD_ref2_best': 'Best ref2 RMSD',
    'TMens': 'TMens',
    'TMmin': 'TMmin',
}


# =============================================================================
# Functions
# =============================================================================


def eval_ensemble(
    sample_fpaths: List[PATH_TYPE], ref1_fpath: PATH_TYPE, ref2_fpath: PATH_TYPE,
    compute_struct_quality: bool = True, compute_sample_diversity = True, max_pairs = 100,
    compute_lddt = False, 
    **align_kwargs
) -> Dict[str, Any]:
    """Evaluate the alignment agianst two reference structures, for one protein

    Args:
        sample_fpaths (List[PATH_TYPE]): a list of sampled conformation pdb files
        ref1_fpath (PATH_TYPE): reference pdb file 1
        ref2_fpath (PATH_TYPE): reference pdb file 2
        compute_struct_quality (bool): if compute structure quality scores for each sample
        compute_sample_diversity (bool): if estimate sample diversity
        max_pairs (int): max pairs to estimate the pairwise diversity between samples
        compute_lddt (bool): if compute lddt scores. Default False.
        **align_kwargs: other kwargs for compute_align_scores

    Returns:
        {
            'metrics': pd.Series of metrics per protein
            'align_scores': pd.DataFrame of alignment to ref1 and ref2
            'struct_scores': pd.DataFrame of structural quality
            'align_pairwise': pd.DataFrame of pairwise alignment
        }
    """
    ret_val = {}
    metrics = []
    
    # -------------------- align to ref --------------------
    align_scores = []
    for sample_path in sample_fpaths:
        result1 = struct_align.compute_align_scores(
            ref_struct=ref1_fpath, test_struct=sample_path,
            compute_lddt=compute_lddt, compute_sc_fape=False,
            show_error=True, **align_kwargs
        )
        result2 = struct_align.compute_align_scores(
            ref_struct=ref2_fpath, test_struct=sample_path,
            compute_lddt=compute_lddt, compute_sc_fape=False,
            show_error=True, **align_kwargs
        )
        align_scores.append({
            'fname': Path(sample_path).stem,
            **{f"{key}_ref1": val for key, val in result1.items()},
            **{f"{key}_ref2": val for key, val in result2.items()}
        })
    
    ret_val['align_scores'] = align_scores = pd.DataFrame(align_scores)
    metrics.append(
        pd.Series({
            'TMscore_ref1_best': align_scores['TMscore_ref1'].max(),
            'TMscore_ref2_best': align_scores['TMscore_ref2'].max(),
            'RMSD_ref1_best': align_scores['RMSD_ref1'].min(),
            'RMSD_ref2_best': align_scores['RMSD_ref2'].min(),
            'TMens': 1/2 * (align_scores['TMscore_ref1'].max() + align_scores['TMscore_ref2'].max()),
            'TMmin': min(align_scores['TMscore_ref1'].max(), align_scores['TMscore_ref2'].max())
        })
    )
    # align ref1 and ref2
    align_ref12 = struct_align.compute_align_scores(
        ref_struct=ref1_fpath, test_struct=ref2_fpath,
        compute_lddt=compute_lddt, compute_sc_fape=False, show_error=True,
        **align_kwargs
    )
    align_ref12 = pd.Series(align_ref12)
    metrics.append(pd.Series({f"{key}_ref12": val for key, val in align_ref12.items()}))

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
            compute_sc_fape=False,
            n_proc=1,
            **align_kwargs
        )
        ret_val['align_pairwise'] = res['align_pairwise']
        metrics.append(res['metrics'])

    ret_val['metrics'] = pd.concat(metrics)
    return ret_val


def report_stats(metrics) -> pd.Series:
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


# =============================================================================
# Plotting
# =============================================================================

def scatterplot_TMscore_ref1_vs_ref2(result, chain_name, ax=None, save_to=None, **kwargs):
    """Scatterplot of TMscore to ref 1 vs ref 2"""
    align_scores = result['align_scores']
    df = align_scores[align_scores['chain_name'] == chain_name]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    else:
        fig = plt.gcf()
    
    _default_kwargs = dict(s=10, alpha=0.8)
    _default_kwargs.update(kwargs)
    sns.scatterplot(data=df, x='TMscore_ref1', y='TMscore_ref2', ax=ax, zorder=3, **_default_kwargs)
    
    metrics = result['metrics']
    TMscore_ref12 = metrics.set_index('chain_name').loc[chain_name, 'TMscore_ref12']
    ax.plot((TMscore_ref12, TMscore_ref12), (0, 1), 'k--', alpha=0.5, lw=0.8, zorder=1)
    ax.plot((0, 1), (TMscore_ref12, TMscore_ref12), 'k--', alpha=0.5, lw=0.8, zorder=1)

    ax.set_title(f"{chain_name} ({len(df)})")
    ax.set_xlabel('TMscore to ref1')
    ax.set_ylabel('TMscore to ref2')
    if save_to is not None:
        fig.tight_layout()
        fig.savefig(save_to, bbox_inches='tight', dpi=300)
    return ax


def scatterplot_TMens_vs_TMref12(result, x='TMscore_ref12', y='TMens', ax=None, save_to=None, **kwargs):
    """TMens vs TM12 scatterplot with baseline y = 0.5 + 0.5 * (x - 0.5)"""
    metrics = result['metrics']
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    else:
        fig = plt.gcf()
    _default_kwargs = dict(s=10)
    _default_kwargs.update(kwargs)
    sns.scatterplot(data=metrics, x='TMscore_ref12', y='TMens', ax=ax, **_default_kwargs)
    ax.set_xlabel('TM$_\mathregular{conf1/conf2}$')
    ax.set_ylabel('TM$_\mathregular{ens}$')
    ax.plot((0,1),(0.5,1), c='gray')
    
    if save_to is not None:
        fig.tight_layout()
        fig.savefig(save_to, bbox_inches='tight', dpi=300)
    return ax

