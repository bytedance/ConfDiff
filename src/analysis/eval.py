"""Evaluation function during for val_gen training

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Literal, Dict, Tuple, Optional
from pathlib import Path, PosixPath

from src.analysis import cameo2022


# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

def eval_gen_conf(
    output_root,
    csv_fpath, 
    ref_root,
    num_samples: Optional[int] = None,
    n_proc=8,
    **align_kwargs
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate generated conformations

    Returns:
        a list of dict contain evaluation metrics
    """
    if num_samples is not None:
        num_samples = int(num_samples)
    output_root: PosixPath = Path(output_root)

    results = cameo2022.eval_cameo2022(
        result_root=output_root,
        metadata_csv_path=csv_fpath,
        ref_root=ref_root, 
        num_samples=num_samples, 
        n_proc=n_proc,
        return_all=False,
        return_wandb_log=True,
        **align_kwargs
    )
    
    log_stats = results.get('log_stats', {})
    log_dists = results.get('log_dists', {})
    return log_stats, log_dists


