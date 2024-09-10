"""Analysis moduel for apo-holo

----------------
[License]

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Union, Optional, Dict, Any, Tuple, Dict, List

import pandas as pd
from pathlib import Path, PosixPath
from src.analysis import dual_ref, struct_quality, struct_diversity
from src.analysis.dual_ref import scatterplot_TMens_vs_TMref12, scatterplot_TMscore_ref1_vs_ref2
from src.utils.misc.process import mp_imap_unordered
from src.utils.hydra_utils import get_pylogger
logger = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================
try:
    from src.utils.misc.env import APO_METADATA_PATH
except:
    APO_METADATA_PATH = None


PATH_TYPE = Union[str, PosixPath]


REPORT_TABLE_COLS = [
    'Num. of cases',
    'TMens',
    'TMmin',
    'Best ref1 TMscore',
    'Best ref2 TMscore',
    'Best ref1 RMSD',
    'Best ref2 RMSD',
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
    ref1_fpath,
    ref2_fpath,
    num_samples=None,
    **align_kwargs
):
    """Evalute metrics for one Apo protein

    Args:
        chain_name (str): protein chain name
        result_dir (path-like): folder path contains the sample pdb files of given protein
        ref1_fapth (path-like): apo pdb path
        ref1_fapth (path-like): holo pdb path
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
        for fpath in Path(result_dir).rglob(f'{chain_name[:4]}*.pdb') 
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

    results = dual_ref.eval_ensemble(
        sample_fpaths=sample_fpaths, ref1_fpath=ref1_fpath, ref2_fpath=ref2_fpath,
        compute_struct_quality=True, compute_sample_diversity=True, max_pairs=100, 
        compute_lddt=False,
        **align_kwargs
    )

    for metric_type, metric_val in results.items():
        if isinstance(metric_val, pd.DataFrame):
            metric_val.insert(0, 'chain_name', chain_name)
        else:
            assert isinstance(metric_val, pd.Series)
            results[metric_type] = pd.concat([pd.Series({'chain_name': chain_name}), metric_val])
    return results


def _work_fn_apo(row, result_root, ref_root, num_samples, **align_kwargs):

    def convert_to_pdb_name(chain_name):
        pdb_id, model_id, chain_id = chain_name.split('_')
        return f"{pdb_id.upper()}_{model_id}_{chain_id}"

    chain_name = row['chain_name']
    apo_chain_name = convert_to_pdb_name(row['apo_chain_name'])
    holo_chain_name = convert_to_pdb_name(row['holo_chain_name'])
    result_dir = Path(f'{result_root}/{chain_name[:4].lower()}_{chain_name[-1]}')
    if not result_dir.exists():
        result_dir = Path(f'{result_root}/{chain_name[:-2].lower()}_{chain_name[-1]}')

    return eval_one_case(
        chain_name, 
        result_dir=result_dir,
        ref1_fpath=Path(ref_root)/apo_chain_name[1:3]/apo_chain_name[:4]/f"{apo_chain_name}.pdb",
        ref2_fpath=Path(ref_root)/holo_chain_name[1:3]/holo_chain_name[:4]/f"{holo_chain_name}.pdb",
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
    report_tab = dual_ref.report_stats(metrics)

    # get summary from struct quality
    sq_report_tab = struct_quality.report_stats(metrics)

    # get diversity stats
    div_report_tab = struct_diversity.report_stats(metrics)

    report_tab = pd.concat([
        pd.Series({'Num. of cases': len(metrics)}), 
        report_tab, 
        sq_report_tab, 
        div_report_tab
    ])

    return pd.Series(report_tab)[REPORT_TABLE_COLS]


def eval_apo(
    result_root: Union[PATH_TYPE, Dict[str, PATH_TYPE]],
    ref_root: PATH_TYPE,
    metadata_csv_path=APO_METADATA_PATH,
    num_samples=None,
    n_proc: int = 1,
    print_not_found = True,
    return_all: bool = False,
    **align_kwargs
):
    if isinstance(result_root, dict):
        # multiple experiments to compare
        ret_val = {}
        report_tab = {}
        for exp_name, exp_result_root in result_root.items():
            exp_ret_val = eval_apo(
                result_root=exp_result_root, metadata_csv_path=metadata_csv_path, ref_root=ref_root,
                num_samples=num_samples, n_proc=n_proc, print_not_found=print_not_found, return_all=return_all,
                **align_kwargs
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
            row_list = [row for _, row in metadata_csv.iterrows()]
            
            result_list = mp_imap_unordered(
                iter=row_list, func=_work_fn_apo, n_proc=n_proc, 
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
        
        ret_val['report_tab'] = report_stats(metrics)
        return ret_val
