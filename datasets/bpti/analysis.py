"""Methods to analyze BPTI results

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Tuple, Dict, Any, List, Union

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path, PosixPath

from src.utils.protein.protein_io import split_pdb_models
from src.analysis import struct_diversity, struct_quality, md_dist
from ..fastfold.analysis import (
    eval_ensemble,
    plot_tica_density_plot
)
from .tica_fit import get_tica_model

# =============================================================================
# Constants
# =============================================================================

PATH_TYPE = Union[str, PosixPath]


METRIC_FORMATTER = {
    'js_pwd': '{:.2f}',
    'js_tic': '{:.2f}',
    'js_rg': '{:.2f}',
    'js_tic_2d': '{:.2f}',
    'contact_rmsd': '{:.2f}',
    'avg_pw_RMSD': '{:.1f}',
    'avg_pw_TMscore': '{:.2f}',
}


METRIC_TABLE_NAME = {
    'js_pwd': 'JS-PwD',
    'js_tic': 'JS-TIC',
    'js_rg': 'JS-Rg',
    'js_tic_2d': 'JS-TIC2D',
    'contact_rmsd': 'RMSE-contact',
    'avg_pw_RMSD': 'pwRMSD',
    'avg_pw_TMscore': 'pwTMscore',
}


REPORT_TABLE_COLS = [
    'pwTMscore',
]

# =============================================================================
# Functions
# =============================================================================


def eval_bpti(
    result_root: Union[PATH_TYPE, Dict[str, PATH_TYPE]],
    dataset_root: PATH_TYPE, metastates_pdb_root: PATH_TYPE,
    num_samples=1000, rerun=False, tica_model='best', 
    n_proc=12,
    **align_kwargs
):
    """Analyze BPTI results"""
    if isinstance(result_root, dict):
        # multiple experiments to compare
        ret_val = {}
        report_tab = {}
        for exp_name, exp_result_root in result_root.items():
            exp_ret_val = eval_bpti(
                result_root=exp_result_root, 
                dataset_root=dataset_root,
                tica_model=tica_model,
                metastates_pdb_root=metastates_pdb_root,
                num_samples=num_samples, 
                **align_kwargs
            )
            report_tab[exp_name] = exp_ret_val['report_tab']
            ret_val[exp_name] = exp_ret_val
        ret_val['report_tab'] = pd.DataFrame(report_tab).T
        return ret_val
    else:
        # -------------------- Set up IO --------------------
        prot_name = 'BPTI'
        chain_name = 'bpti'
        result_root: PosixPath = Path(result_root)
        output_root = result_root/'results'
        result_root = result_root/chain_name
        dataset_root: PosixPath = Path(dataset_root)

        # -------------------- MD traj analysis --------------------
        tica_model: Dict = get_tica_model(lagtime=tica_model, dataset_root=dataset_root)

        if result_root.joinpath(f'{chain_name}.pdb').exists() and \
            not output_root.parent.joinpath(f'{chain_name}_sample{num_samples - 1}.pdb').exists():
            # single pdb files contain multiple models, split PDB file
            result_root = split_pdb_models(result_root/f'{chain_name}.pdb', output_dir=output_root.parent)

        sample_fpath_list = [fpath for fpath in result_root.glob(f'{chain_name}*sample*.pdb') if '_conf.pdb' not in fpath.name and 'traj' not in fpath.name]
        print(f"{len(sample_fpath_list)} sample found.")
        if num_samples is None:
            print(f"[{chain_name}] {len(sample_fpath_list)} samples found")
        elif num_samples != len(sample_fpath_list):
            print(f'[{chain_name}] sample number mismatch: {len(sample_fpath_list)} found vs {num_samples} expected')
        
        # Get reference pdb
        metastat_pdb_list = list(Path(metastates_pdb_root).glob('*.pdb'))
    
        ret_val = eval_ensemble(
            sample_fpaths=sample_fpath_list, output_root=output_root, 
            ref_val_root=dataset_root/'fullmd_ref_value',
            tica_model=tica_model,
            ref_pdbs=metastat_pdb_list,
            rerun=rerun,
            compute_struct_quality=True,
            compute_sample_diversity=True, max_pairs=100,
            n_proc=n_proc,
            **align_kwargs,
        )

        # -------------------- Plots --------------------
        if rerun or not output_root.joinpath('scatterplot-tica.png').exists():
            plot_tica_density_plot(
                tic_proj=ret_val['tica_proj'], ref_tic_dist=ret_val['ref_tica_dist_2d'], ax=None
            )
            ax_tica = plt.gca()
            fig = plt.gcf()
            ax_tica.set_title(f"{prot_name} (hit: {ret_val['metrics']['hit_tic_2d']:.1%}, recovery: {ret_val['metrics']['rec_tic_2d']:.1%})", fontsize=12)
            fig.savefig(str(output_root/'scatterplot-tica.png'))
            plt.close()
        
        if rerun or not output_root.joinpath('contact-map.png').exists():
            md_dist.plot_contact_map(
                contact_rate=ret_val['contact_rate'], min_contact_rate=1e-3, log_scale=True, ax=None
            )
            ax_contactmap = plt.gca()
            fig = plt.gcf()
            ax_contactmap.set_title(f"{prot_name}", fontsize=12)
            fig.savefig(str(output_root/'contact-map.png'))
            plt.close()
        
        # -------------------- Summarize report tab --------------------
        report_tab = {}
        metrics = ret_val['metrics']

        # MD metrics + diveristy metrics
        metric_formatter = METRIC_FORMATTER
        metric_tab_name = METRIC_TABLE_NAME
        for metric_name in metric_formatter.keys():
            if metric_name not in metrics.index:
                continue
            formatter = metric_formatter[metric_name]
            tab_name = metric_tab_name[metric_name]
            report_tab[tab_name] = formatter.format(metrics[metric_name])
        
        report_tab = pd.Series(report_tab)[REPORT_TABLE_COLS]
        # Compare to metastable states
        for ref_fpath in metastat_pdb_list:
            report_tab[f'Best RMSD ({ref_fpath.stem})'] = metrics[f'min_RMSD_{ref_fpath.stem}']
        report_tab['RMSDens'] = report_tab[[f'Best RMSD ({ref_fpath.stem})' for ref_fpath in metastat_pdb_list]].mean()
        
        return dict(metrics=metrics, report_tab=report_tab, all_results=ret_val)
