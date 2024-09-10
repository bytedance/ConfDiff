"""Analysis module for CAMEO2022

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Union, Optional, Dict, Any, Tuple, Dict, List

import pandas as pd

from pathlib import Path, PosixPath

from src.analysis import single_ref, struct_quality, struct_diversity
from src.utils.misc.process import mp_imap_unordered

from src.utils.hydra_utils import get_pylogger
logger = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================

PATH_TYPE = Union[str, PosixPath]


METRIC_FORMATTER = {
    'avg_RMSD': '{:.2f}',
    'avg_TMscore': '{:.3f}',
    'avg_GDT-TS': '{:.3f}',
    'avg_GDT-HA': '{:.3f}',
    'avg_lDDT': '{:.3f}',
    'best_RMSD': '{:.2f}',
    'best_TMscore': '{:.3f}',
    'best_GDT-TS': '{:.3f}',
    'best_GDT-HA': '{:.3f}',
    'best_lDDT': '{:.3f}',
}


REPORT_TABLE_COLS = [
    'Num. of cases',
    'TMscore (best)',
    'RMSD (best)',
    'GDT-TS (best)',
    'GDT-HA (best)',
    'lDDT (best)',
    'TMscore (avg)',
    'RMSD (avg)',
    'GDT-TS (avg)',
    'GDT-HA (avg)',
    'lDDT (avg)',
    'pwTMscore',
    'pwRMSD',
    'Secondary Structure Rate',
    'Strand Rate',
    'Broken C=N bond'
]


# =============================================================================
# Functions
# =============================================================================

def eval_one_case(
    chain_name,
    result_dir,
    ref_fpath,
    num_samples=None,
    **align_kwargs
):
    """Evaluate metrics for one CAMEO protein

    Args:
        chain_name (str): protein chain name
        result_dir (path-like): folder path contains the sample pdb files of given protein
        ref_fapth (path-like): reference pdb path
        num_samples (int, optional): number of samples to evaluate
        **align_kwargs

    Returns:
        {
            'metrics': pd.Series of metrics per protein
            'align_scores': pd.DataFrame of alignment to ref struct. 
            'struct_scores': pd.DataFrame of structural quality
            'align_pairwise': pd.DataFrame of pairwise alignment
        }

    """
    result_dir = Path(result_dir)
    sample_fpaths = [
        fpath 
        for fpath in Path(result_dir).rglob(f'{chain_name}*.pdb') 
        if 'traj' not in fpath.name
    ]

    if len(sample_fpaths) == 0:
        logger.warn(f"[{chain_name}] no sample found")
        return chain_name

    if num_samples is not None:
        if len(sample_fpaths) < num_samples:
            logger.warn(f"[{chain_name}] only found samples: {len(sample_fpaths)} < {num_samples}")
        elif len(sample_fpaths) > num_samples:
            sample_fpaths = sample_fpaths[:num_samples]

    results = single_ref.eval_ensemble(
        sample_fpaths=sample_fpaths, ref_fpath=ref_fpath, 
        compute_lddt=True, compute_sc_fape=False,
        compute_struct_quality=True, compute_sample_diversity=True,
        **align_kwargs
    )

    for metric_type, metric_val in results.items():
        if isinstance(metric_val, pd.DataFrame):
            metric_val.insert(0, 'chain_name', chain_name)
        else:
            assert isinstance(metric_val, pd.Series)
            results[metric_type] = pd.concat([pd.Series({'chain_name': chain_name}), metric_val])
    return results


def _work_fn_cameo2022(chain_name, result_root, ref_root, num_samples, **align_kwargs):
    pdb_id, chain_id = chain_name.split('_')
    pdb_chain_name = f"{pdb_id.upper()}_0_{chain_id}"
    return eval_one_case(
        chain_name, 
        result_dir=Path(result_root)/chain_name,
        ref_fpath=Path(ref_root)/pdb_chain_name[1:3]/f"{pdb_chain_name}.pdb",
        num_samples=num_samples,
        **align_kwargs
    )


def report_stats(metrics: pd.DataFrame) -> pd.Series:
    """Summarize metrics and generate report table

    It summarizes over all proteins

    Returns:
        report_tab (pd.Series): summarized report table row
    """

    # Get summary from single_ref
    report_tab = single_ref.report_stats(metrics)

    # get summary from struct quality
    sq_report_tab = struct_quality.report_stats(metrics)

    # get diversity stats
    div_report_tab = struct_diversity.report_stats(metrics)

    report_tab = pd.concat([pd.Series({'Num. of cases': len(metrics)}), report_tab, sq_report_tab, div_report_tab])

    return pd.Series(report_tab)[REPORT_TABLE_COLS]


def get_wandb_log(metrics):
    """Get metrics stats for WANDB logging"""
    log_stats = {}
    log_dists= {}
    for metric_name in METRIC_FORMATTER.keys():
        if metric_name not in metrics.columns:
            continue 
        log_stats[f"{metric_name}_mean"] = metrics[metric_name].mean()
        log_stats[f"{metric_name}_median"] = metrics[metric_name].median()
        log_dists[metric_name] = list(metrics[metric_name].values)

    # get summary from struct quality
    sq_log_stats, sq_log_dist = struct_quality.get_wandb_log(metrics)
    log_stats.update(sq_log_stats)
    log_dists.update(sq_log_dist)

    # get diversity stats
    div_log_stats, div_log_dist = struct_diversity.get_wandb_log(metrics)
    log_stats.update(div_log_stats)
    log_dists.update(div_log_dist)

    return log_stats, log_dists


def eval_cameo2022(
    result_root: Union[PATH_TYPE, Dict[str, PATH_TYPE]],
    ref_root: PATH_TYPE,
    metadata_csv_path: PATH_TYPE,
    num_samples=None,
    n_proc: int = 1,
    print_not_found = True,
    return_all: bool = False,
    return_wandb_log: bool = False,
    **align_kwargs
):
    """Evaluate CAMEO2022 result.

    Args:
        result_root (PATH_TYPE): path to generated samples or a dictionary of {exp: result_root}
        metadata_csv_path (PATH_TYPE): path to cameo2022 metadata csv
        ref_root (PATH_TYPE): path to reference PDB file root
        num_samples (int): check the number of samples for each protein, optional
        n_proc (int): number of parallel processes
        print_not_found (bool): if print missing results
        return_all (bool): if return all metrics, including alignment and quality metrics for each generated conformation.
        return_wandb_log (bool): if return stats for WANDB logging. Used by val_gen during training
        **align_kwargs: other kwargs pass to compute_align_scores
    
    Returns:
        Dict or a dict of:
        {
            'metrics': metrics summary per protein
            'report_tab': report table (simplified) per experiment
            and other metrics
        }
    """
    if isinstance(result_root, dict):
        # multiple experiments to compare
        ret_val = {}
        report_tab = {}
        for exp_name, exp_result_root in result_root.items():
            exp_ret_val = eval_cameo2022(
                result_root=exp_result_root, metadata_csv_path=metadata_csv_path, ref_root=ref_root,
                num_samples=num_samples, n_proc=n_proc, print_not_found=print_not_found, return_all=return_all,
                return_wandb_log = False, **align_kwargs
            )
            report_tab[exp_name] = exp_ret_val['report_tab']
            ret_val[exp_name] = exp_ret_val
        ret_val['report_tab'] = pd.DataFrame(report_tab).T
        return ret_val
    else:
        # single experiment
        result_root: PosixPath = Path(result_root)
        assert result_root.exists(), f"result_root not found: {result_root}"
        ref_root = Path(ref_root)
        assert ref_root.exists(), f"ref_root not found: {ref_root}"
        output_root = result_root/'results'
        output_root.mkdir(parents=True, exist_ok=True)

        ret_val: Dict[str, Any] = {'output_root': output_root}

        if output_root.joinpath('metrics_summary.csv').exists():
            # Load precomputed results
            print(f'Load metrics from {result_root} ...')
            metrics = ret_val['metrics'] = pd.read_csv(output_root/'metrics_summary.csv')
            if return_all:
                for submetric in ['align_scores', 'struct_scores', 'align_pairwise']:
                    if output_root.joinpath(submetric + '.csv').exists():
                        ret_val[submetric] = pd.read_csv(output_root/f"{submetric}.csv")
        else:
            # Run evaluation
            metadata_csv = pd.read_csv(metadata_csv_path)
            print(f'Evaluating {Path(metadata_csv_path).name} for {result_root} ...')
            chain_name_list = metadata_csv['chain_name'].tolist()
            
            result_list = mp_imap_unordered(
                iter=chain_name_list, func=_work_fn_cameo2022, n_proc=n_proc, 
                result_root=result_root, ref_root=ref_root, num_samples=num_samples, **align_kwargs
            )
            valid_results = [result for result in result_list if not isinstance(result, str)]
            not_found = [result for result in result_list if isinstance(result, str)]
            if len(not_found) > 0 and print_not_found:
                print("Following results are not found: " + ' '.join(not_found))
            
            # post process
            if len(valid_results) == []:
                print(f"No results found under {result_root}")
                return {}
            else:
                # save and summarize results
                metrics = ret_val['metrics'] = pd.DataFrame([results['metrics'] for results in valid_results])
                metrics.to_csv(output_root/'metrics_summary.csv', index=False)
                for submetric in ['align_scores', 'struct_scores', 'align_pairwise']:
                    all_scores = pd.concat(
                        [results[submetric] for results in valid_results], 
                        ignore_index=True
                    )
                    all_scores.to_csv(output_root/f'{submetric}.csv', index=False)
                    if return_all:
                        ret_val[submetric] = all_scores
        
        if return_wandb_log:
            ret_val['log_stats'], ret_val['log_dist'] = get_wandb_log(metrics)
        else:
            ret_val['report_tab'] = report_stats(metrics)
        return ret_val
