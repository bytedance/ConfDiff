"""Methods to analyze fast-folding results

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Tuple, Dict, Any, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path, PosixPath

import mdtraj
from deeptime.decomposition import TICA
from src.analysis.md_dist import (
    hist_contour_plot, 
    scatterplot_with_density, 
    plot_contact_map,
    compute_js_pwd,
    compute_js_tic_1d,
    compute_js_tic_2d,
    compute_js_rg,
    compute_contact_rate
)
from src.analysis.mdtraj_tools import (
    get_CA_pairwise_dist, 
    get_radius_of_gyration, 
    filter_CA_pairwise_dist,
)
from src.utils.protein.protein_io import split_pdb_models
from src.analysis import struct_quality, struct_align, struct_diversity
from datasets.fastfold.tica_fit import get_tica_model
from src.utils.misc import process


# =============================================================================
# Constants
# =============================================================================

from datasets.fastfold.info import CHAIN_NAME_TO_PROT_NAME, TICA_LAGTIME, DATASET_DIR, PRECOMPUTED_REF_DIR, DEFAULT_METADATA


PATH_TYPE = Union[str, PosixPath]


METRIC_FORMATTER = {
    'js_pwd': '{:.2f}',
    'js_tic': '{:.2f}',
    'js_rg': '{:.2f}',
    'js_tic_2d': '{:.2f}',
    'contact_rmsd': '{:.2f}',
}


METRIC_TABLE_NAME = {
    'js_pwd': 'JS-PwD',
    'js_tic': 'JS-TIC',
    'js_rg': 'JS-Rg',
    'js_tic_2d': 'JS-TIC2D',
    'contact_rmsd': 'RMSE-contact',
}


REPORT_TABLE_COLS = [
    'Num. of cases',
    'JS-PwD',
    'JS-Rg',
    'JS-TIC',
    'JS-TIC2D',
    'Val-Clash (CA)',
    'Val-Break (CA)',
    'Val-CA',
    'pwTMscore'
]

# =============================================================================
# Functions
# =============================================================================


def eval_ensemble(
    sample_fpaths, output_root,
    ref_val_root, tica_model: Dict, 
    ref_pdbs=None,
    compute_struct_quality=True,
    compute_sample_diversity=True, max_pairs=100,
    rerun=False, n_proc=1, **align_kwargs,
) -> Dict[str, Any]:
    """Eval the ensemble result for one protein"""

    ret_val = {}
    output_root.mkdir(parents=True, exist_ok=True)
    ref_tica_dist_1d = tica_model['dist_1d']
    ret_val['ref_tica_dist_2d'] = ref_tica_dist_2d = tica_model['dist_2d']
    tica_model = tica_model['model'].fetch_model()

    if rerun or not output_root.joinpath('metrics.csv').exists():
        # PROCESS RESULTS
        
        # -------------------- Compute feats --------------------
        output_root.mkdir(exist_ok=True)

        # Load ensembles
        ensemble = mdtraj.load([str(fpath) for fpath in sample_fpaths]) # mdtraj should do A -> nm automatically
        CA_atom_idx = ensemble.top.select('name CA')
        ensemble.atom_slice(CA_atom_idx, inplace=True)
        n_frames = ensemble.n_frames
        seqlen = ensemble.n_atoms

        # CA pairwise distance
        pdist, pair_idx = get_CA_pairwise_dist(ensemble, excluded_neighbors=0) # get all pairs
        assert pdist.shape == (n_frames, seqlen * (seqlen - 1) / 2), \
            f'pdist shape mismatch: {pdist.shape} for {seqlen} atoms {n_frames} frames'
        assert pair_idx.shape == (seqlen * (seqlen - 1) / 2, 2), \
            f'pair_idx shape mismatch: {pair_idx.shape} for {seqlen} atoms {n_frames} frames'
        np.savez_compressed(output_root/'pdist.npz', pdist=pdist, pair_idx=pair_idx)

        # radius of gyration (rg)
        rg = get_radius_of_gyration(ensemble)
        assert rg.shape == (n_frames, seqlen), \
            f'rg shape mismatch: {rg.shape} for {seqlen} atoms {n_frames} frames'
        np.save(output_root/'rg.npy', rg)

        # contact map
        contact_rate = ret_val['contact_rate'] = compute_contact_rate(pdist, pair_idx, seqlen)
        np.save(output_root/'contact_rate.npy', contact_rate)
        
        # TICA projection
        tica_proj = ret_val['tica_proj'] = tica_model.transform(pdist)[:, :2]
        np.save(output_root/'tica_proj.npy', tica_proj)

        # -------------------- Evaluation --------------------
        # Dist similarity: JS-PwD, JS-TIC, JS-Rg
        ref_pwd_dist = ret_val['ref_pwd_dist'] = dict(np.load(ref_val_root/'pwd_dist.npz'))
        
        js_pwd, hit_pwd = compute_js_pwd(pdist, pair_idx, ref_pwd_dist, n_proc=n_proc)
        js_tic_2d, hit_tic_2d, rec_tic_2d, precision_2d, recall_2d, f1_2d = compute_js_tic_2d(tica_proj, ref_tica_dist_2d)
        js_tic_1d, hit_tic_1d = compute_js_tic_1d(tica_proj, ref_tica_dist_1d, n_proc=n_proc)
        
        ref_rg_dist = ret_val['ref_rg_dist'] = dict(np.load(ref_val_root/'rg_dist.npz'))
        js_rg, hit_rg = compute_js_rg(rg, ref_rg_dist, n_proc=n_proc)

        ref_contact_rate = ret_val['ref_contact_rate'] = np.load(ref_val_root/"contact_map.npy")
        row, col = np.triu_indices_from(ref_contact_rate, k=1)
        contact_rmsd = np.sqrt(np.sum((ref_contact_rate - contact_rate)[row, col] ** 2))

        metrics = [pd.Series({
            'seqlen': seqlen,
            'n_sample': n_frames,
            'contact_rmsd': contact_rmsd,
            'js_pwd': js_pwd, 'hit_pwd': hit_pwd,
            'js_tic': js_tic_1d, 'hit_tic': hit_tic_1d,
            'js_tic_2d': js_tic_2d, 'hit_tic_2d': hit_tic_2d, 'rec_tic_2d': rec_tic_2d, 'tic_prec_2d': precision_2d, 'tic_recall_2d': recall_2d, 'tic_f1_2d': f1_2d,
            'js_rg': js_rg, 'hit_rg': hit_rg,
        })]

        # -------------------- Conf quality --------------------
        # struct quality
        if compute_struct_quality:
            res = struct_quality.eval_ensemble(sample_fpaths)
            struct_scores = ret_val['struct_scores'] = res['struct_scores']
            metrics.append(res['metrics'])
            struct_scores.to_csv(output_root/'struct_scores.csv', index=False)

        if ref_pdbs is not None:
            # distance to all ref structures
            align_scores = ret_val['align_scores'] = struct_align.align_to_group(
                sample_fpaths, ref_pdbs, n_proc=n_proc, 
                compute_lddt=False, compute_sc_fape=False, **align_kwargs
            )
            align_scores.to_csv(output_root/'align_scores.csv')
            group_align_metrics = pd.Series({
                'RMSD_to_ref': align_scores['min_RMSD'].mean(),
                'TMscore_to_ref': align_scores['max_TMscore'].mean()
            })
            for ref_fpath in ref_pdbs:
                group_align_metrics[f'max_TMscore_{ref_fpath.stem}'] = align_scores[f'TMscore_{ref_fpath.stem}'].max()
                group_align_metrics[f'min_RMSD_{ref_fpath.stem}'] = align_scores[f'RMSD_{ref_fpath.stem}'].min()
            metrics.append(group_align_metrics)
        if compute_sample_diversity:
            res = struct_diversity.eval_ensemble(
                sample_fpaths, max_samples=max_pairs, 
                compute_lddt=False, 
                compute_sc_fape=False,
                n_proc=n_proc,
                **align_kwargs
            )
            align_pairwise = ret_val['align_pairwise'] = res['align_pairwise']
            align_pairwise.to_csv(output_root/'align_pairwise.csv', index=False)
            metrics.append(res['metrics'])
        metrics = pd.concat(metrics)
        pd.DataFrame([metrics]).to_csv(output_root/'metrics.csv', index=False)
        ret_val['metrics'] = metrics
    else:
        # Load precomputed results
        ret_val['pdist'] = dict(np.load(output_root/'pdist.npz'))
        ret_val['rg'] = np.load(output_root/'rg.npy')
        ret_val['contact_rate'] = np.load(output_root/'contact_rate.npy')
        ret_val['tica_proj'] = np.load(output_root/'tica_proj.npy')
        if compute_struct_quality:
            ret_val['struct_scores'] = pd.read_csv(output_root/'struct_scores.csv', index_col=None)
        if ref_pdbs is not None:
            ret_val['align_scores'] = pd.read_csv(output_root/'align_scores.csv', index_col=None)
        if compute_sample_diversity:
            align_pairwise = pd.read_csv(output_root/'align_pairwise.csv', index_col=None)
        ret_val['metrics'] = pd.read_csv(output_root/'metrics.csv').squeeze()

        # load ref
        ret_val['ref_pwd_dist'] = dict(np.load(ref_val_root/'pwd_dist.npz'))
        ret_val['ref_rg_dist'] = dict(np.load(ref_val_root/'rg_dist.npz'))
        ret_val['ref_contact_rate'] = np.load(ref_val_root/"contact_map.npy")

    return ret_val


def plot_tica_density_plot(
    tic_proj, ref_tic_dist=None,
    ax=None, max_kde_sample=1000, max_sample_show=200, scatter_sample_order=0.5, contour_kwargs=None,
    scale=False,
    **kwargs
) :
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    if ref_tic_dist is not None:
        # contour plot in the back
        H, xedges, yedges = ref_tic_dist['H'], ref_tic_dist['xedges'], ref_tic_dist['yedges']
        _default_contour_kwargs = dict(cmap='GnBu', zorder=1)
        if contour_kwargs is not None:
            _default_contour_kwargs.update(contour_kwargs)
        hist_contour_plot(H=H, xedges=xedges, yedges=yedges, ax=ax, **_default_contour_kwargs)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    else:
        xlim = ylim = None

    # density estimated scatter plot over the top
    scatterplot_with_density(
        data=tic_proj, ax=ax, cmap='OrRd', s=30, zorder=3,
        max_kde_sample=max_kde_sample, max_sample_show=max_sample_show, scatter_sample_order=scatter_sample_order,
        **kwargs
    )
    if not scale and xlim is not None:
        # Don't scale
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
 
    return ax


def eval_one_protein(
    chain_name, result_root, ref_root, num_samples=1000, tica_model='best', 
    n_proc=1,
    rerun=False, 
    **align_kwargs,
) -> Dict[str, Any]:
    """Analyze sampled conformations for one fast-folding protein
    """
    # -------------------- setup IO --------------------
    output_root = Path(result_root)/'results'/chain_name
    result_root = Path(result_root)/chain_name

    # -------------------- MD traj analysis --------------------
    prot_name = CHAIN_NAME_TO_PROT_NAME[chain_name]
    tica_model = get_tica_model(chain_name, lagtime=tica_model)

    if result_root.joinpath(f'{chain_name}.pdb').exists() and \
        not output_root.parent.joinpath(f'{chain_name}_sample{num_samples - 1}.pdb').exists():
        # single pdb files contain multiple models, split PDB file
        result_root = split_pdb_models(result_root/f'{chain_name}.pdb', output_dir=output_root.parent)
        # print(f"Split pdb files from {sample_root/f'{chain_name}.pdb'} to {sample_root}")
    sample_fpath_list = [fpath for fpath in result_root.glob(f'{chain_name}*sample*.pdb') if '_conf.pdb' not in fpath.name and 'traj' not in fpath.name]
    if num_samples is None:
        print(f"[{chain_name}] {len(sample_fpath_list)} samples found")
    elif num_samples != len(sample_fpath_list):
        print(f'[{chain_name}] sample number mismatch: {len(sample_fpath_list)} found vs {num_samples} expected')
    
    ret_val = eval_ensemble(
        sample_fpaths=sample_fpath_list, output_root=output_root, 
        ref_val_root=Path(ref_root)/chain_name,
        tica_model=tica_model,
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
        plot_contact_map(
            contact_rate=ret_val['contact_rate'], min_contact_rate=1e-3, log_scale=True, ax=None
        )
        ax_contactmap = plt.gca()
        fig = plt.gcf()
        ax_contactmap.set_title(f"{prot_name}", fontsize=12)
        fig.savefig(str(output_root/'contact-map.png'))
        plt.close()

    return ret_val


def report_stats(metrics: pd.DataFrame) -> pd.Series:
    """Summarize metrics and generate report table

    It summarizes over all proteins

    Returns:
        report_tab (pd.Series): summarized report table row
    """
    # Get summary 
    report_tab: Dict[str, Any] = {'Num. of cases': len(metrics)}
    metric_formatter = METRIC_FORMATTER
    metric_tab_name = METRIC_TABLE_NAME
    for metric_name in metric_formatter.keys():
        if metric_name not in metrics.columns:
            continue
        val_mean = metrics[metric_name].mean()
        val_median = metrics[metric_name].median()

        formatter = metric_formatter[metric_name]
        tab_name = metric_tab_name[metric_name]
        report_tab[tab_name] = formatter.format(val_mean) + '/' + formatter.format(val_median)

    # get summary from struct quality
    sq_report_tab = struct_quality.report_stats(metrics)

    # get diversity stats
    div_report_tab = struct_diversity.report_stats(metrics)

    report_tab = pd.concat([
        pd.Series(report_tab),
        sq_report_tab, 
        div_report_tab
    ])

    return report_tab[REPORT_TABLE_COLS]


def eval_fastfold(
    result_root: Union[PATH_TYPE, Dict[str, PATH_TYPE]],
    ref_root: PATH_TYPE,
    metadata_csv_path = DEFAULT_METADATA,
    num_samples: int = 1000,
    chain_name_list = None,
    n_proc=12,
    **align_kwargs
):
    if isinstance(result_root, dict):
        # multiple experiments to compare
        ret_val = {}
        report_tab = {}
        for exp_name, exp_result_root in result_root.items():
            exp_ret_val = eval_fastfold(
                result_root=exp_result_root, metadata_csv_path=metadata_csv_path, ref_root=ref_root,
                num_samples=num_samples, n_proc=n_proc,
                **align_kwargs
            )
            report_tab[exp_name] = exp_ret_val['report_tab']
            ret_val[exp_name] = exp_ret_val
        ret_val['report_tab'] = pd.DataFrame(report_tab).T
        return ret_val
    else:
        # single experiment
        if chain_name_list is None:
            fastfold_metadata = pd.read_csv(metadata_csv_path)
            chain_name_list = list(fastfold_metadata['chain_name'])
            print(f'Evaluating {Path(metadata_csv_path).name} for {result_root} ...')
        else:
            print(f"Evaluating {len(chain_name_list)} proteins for {result_root} ...")

        results_list = process.mp_imap(
            func=eval_one_protein, iter=chain_name_list, n_proc=n_proc,
            result_root=result_root, ref_root=ref_root, num_samples=num_samples, 
            tica_model='best', **align_kwargs,
        )

        metrics = pd.DataFrame({
            chain_name: results.pop('metrics') for chain_name, results in zip(chain_name_list, results_list)
        }).T
        all_results = {
            chain_name: results for chain_name, results in zip(chain_name_list, results_list)
        }

        report_tab = report_stats(metrics)
        return dict(metrics=metrics, report_tab=report_tab, all_results=all_results)