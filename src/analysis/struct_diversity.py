"""Methods to evaluate structural diveristy

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
import pandas as pd

from itertools import combinations
from pathlib import Path
from random import shuffle

from src.utils.misc.process import mp_imap
from .struct_align import compute_align_scores

# =============================================================================
# Constants
# =============================================================================
from .struct_align import METRIC_FORMATTER

DIV_METRICS_NAME = {
    'avg_pw_RMSD': 'pwRMSD',
    'avg_pw_TMscore': 'pwTMscore',
    'avg_pw_GDT-TS': 'pwGDT-TS',
    'avg_pw_lDDT': 'pwlDDT'
}

# =============================================================================
# Functions
# =============================================================================

def _worker_pw_align(arg, **kwargs):
    """Pairwise alignment worker"""
    conf1_fpath, conf2_fpath = arg
    return compute_align_scores(ref_struct=conf1_fpath, test_struct=conf2_fpath, **kwargs)


def eval_ensemble(fpath_list, max_samples=None, n_proc=None, mute_tqdm=True, **kwargs):
    """Sample pairwise structural alignments between structures in fpath_list
    
    Args:
        fpath_list: list of PDB files
        max_samples: maximum number of pairs to calculate
        n_proc: number of processes to use for sampling
    """
    pairs = list(combinations(fpath_list, r=2))
    ordered_pairs = pairs + [(path2, path1) for path1, path2 in pairs]

    if max_samples is not None and len(ordered_pairs) > max_samples:
        shuffle(ordered_pairs)
        ordered_pairs = ordered_pairs[:max_samples]
    
    align_scores = mp_imap(
        func=_worker_pw_align, iter=ordered_pairs, n_proc=n_proc, mute_tqdm=mute_tqdm,
        **kwargs
    )

    align_scores = pd.DataFrame([
        {
            'fname1': Path(conf1_fpath).name,
            'fname2': Path(conf2_fpath).name,
            **scores
        } for (conf1_fpath, conf2_fpath), scores in zip(ordered_pairs, align_scores)
    ])

    metric_col = [col for col in METRIC_FORMATTER.keys() if col in align_scores.columns]
    metrics = align_scores[metric_col].mean().rename(lambda key: f'avg_pw_{key}')

    return {
        'metrics': metrics,
        'align_pairwise': align_scores
    }


def report_stats(metrics: pd.DataFrame):
    """Get report table for pairwise structural alignment
    
    It summarizes over the multiple proteins
    """
    
    report_tab = {}

    for metric_name, formatter in METRIC_FORMATTER.items():
        metric_name = f"avg_pw_{metric_name}"
        if metric_name in metrics.keys():
            val_mean = metrics[metric_name].mean()
            val_median = metrics[metric_name].median()
            report_tab[DIV_METRICS_NAME[metric_name]] = formatter.format(val_mean) + '/' + formatter.format(val_median)

    return pd.Series(report_tab)


def get_wandb_log(metrics: pd.DataFrame):
    """Get metrics stats for WANDB logging"""

    log_stats = {}
    log_dists = {}

    for metric_name in METRIC_FORMATTER.keys():
        metric_name = f"avg_pw_{metric_name}"
        if metric_name in metrics.keys():
            log_stats[f"{metric_name}_mean"] = metrics[metric_name].mean()
            log_stats[f"{metric_name}_median"] = metrics[metric_name].median()
            log_dists[metric_name] = list(metrics[metric_name].values)

    return log_stats, log_dists
