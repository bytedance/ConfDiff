"""Functions for protein structural analysis

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Dict, List, Union, Literal

import warnings
warnings.filterwarnings("ignore", ".* is deprecated .*")

import os
import numpy as np
import pandas as pd
import mdtraj as md
from pathlib import PosixPath, Path
from itertools import combinations

from openfold.np import residue_constants

from src.utils.protein.seq import res2aacode
from src.utils.protein.protein_io import get_protein_chain, coords_to_npy
from src.utils.hydra_utils import get_pylogger

logger = get_pylogger(__name__)


# =============================================================================
# Constants
# =============================================================================
PATH_TYPE = Union[str, PosixPath]


STRUCT_METRICS_NAME = {
    'CA_clash_validity_str2str': 'Val-Clash (CA)',
    'CA_break_validity_str2str': 'Val-Break (CA)',
    'CA_validity_str2str': 'Val-CA',
    'avg_CNBond_break_rate_biopython': 'Broken C=N bond',
    'avg_ss_percent': 'Secondary Structure Rate',
    'avg_strand_percent': 'Strand Rate'
}

STRUCT_METRICS_FORMATTER = {
    'CA_clash_validity_str2str': '{:.3f}',
    'CA_break_validity_str2str': '{:.3f}',
    'CA_validity_str2str': '{:.3f}',
    'avg_CNBond_break_rate_biopython': '{:.3f}',
    'avg_ss_percent': '{:.3f}',
    'avg_strand_percent': '{:.3f}'
}

# =============================================================================
# Main functions
# =============================================================================


def compute_struct_metrics(pdb_fpath: PATH_TYPE) -> Dict[str, float]:
    """Compute structural metrics on backbone (N, CA, C, O) for a PDB file or a list of PDB files

    Returns:
        metric dictionary or a list of metric dictionaries
    """
    res_aa, res_id, atom_list, backbone_coords = coords_to_npy(
        pdb_fpath, atom_list=('N', 'CA', 'C', 'O')
    )
    ca_coords = backbone_coords[:, 1, :]
    backbone_dist = get_backbone_bond_distance(backbone_coords, resid=res_id, aa=list(res_aa))
    ss_info = compute_secondary_struct(pdb_fpath)

    return dict(
        fname=Path(pdb_fpath).name,
        CA_clash_rate_str2str=compute_CA_clash_rate(ca_coords, tol=0.4),
        CA_disconnect_rate_str2str = compute_CA_disconnect_rate(ca_coords),
        bad_CN_bond_rate_biopython = compute_bad_CN_bond_rate_biopython(backbone_dist['C-N']),
        **ss_info # ss_percent, coil_percent, helix_percent, strand_percent
    )


def eval_ensemble(sample_fpaths):
    """Evaluate the backbone struct quality for an ensemble of conformations
    It calculates following metrics:
        - CA_validity_str2str: fraction of valid structure, using the CA criteria from Str2Str (https://arxiv.org/abs/2306.03117)
        - avg_CA_clash_rate_str2str: clash rate, using the CA criteria from Str2Str
        - CA_clash_validity_str2str: fraction of structures without clashing, from Str2Str
        - avg_CA_break_rate_str2str: break rate, using the CA criteria from Str2Str
        - CA_break_validity_str2str: fraction of structures without breaking, from Str2Str
        - avg_CNBond_break_rate_biopython: rate of broken C=N peptide bonds, from Biopython
        - CNBond_break_validity_biopython: fraction of structures without broken C=N bond
        - avg_ss_percent: average fraction of AA forming secondary structures 
        - avg_strand_percent: average fraction of AA forming strands
    """
    if isinstance(sample_fpaths, (str, PosixPath)):
        sample_fpaths = [sample_fpaths]

    struct_scores = pd.DataFrame([
        {
            'fname': os.path.basename(sample_fpath),
            **compute_struct_metrics(sample_fpath)
        } for sample_fpath in sample_fpaths
    ])

    metrics = pd.Series({
        'CA_validity_str2str': 1 - (struct_scores[['CA_clash_rate_str2str', 'CA_disconnect_rate_str2str']] > 0).any(axis=1).mean(), # composed CA validity
        'avg_CA_clash_rate_str2str': np.mean(struct_scores['CA_clash_rate_str2str']), 
        'CA_clash_validity_str2str': 1 - np.mean(struct_scores['CA_clash_rate_str2str'] > 0),  
        'avg_CA_break_rate_str2str': np.mean(struct_scores['CA_disconnect_rate_str2str']),
        'CA_break_validity_str2str': 1 - np.mean(struct_scores['CA_disconnect_rate_str2str'] > 0),
        'avg_CNBond_break_rate_biopython': np.mean(struct_scores['bad_CN_bond_rate_biopython']),
        'CNBond_break_validity_biopython': 1 - np.mean(struct_scores['bad_CN_bond_rate_biopython'] > 0),
        'avg_ss_percent': np.mean(struct_scores['ss_percent']),
        'avg_strand_percent': np.mean(struct_scores['strand_percent']),
    })

    return {
        'metrics': metrics,
        'struct_scores': struct_scores
    }


def report_stats(metrics):
    """Get report table on backbone struct quality.

    It summarizes over the multiple proteins
    """
    report_tab = {}
    for metric_name in STRUCT_METRICS_NAME:
        if metric_name not in metrics.columns:
            continue

        val_mean = metrics[metric_name].mean()
        val_median = metrics[metric_name].median()

        formatter = STRUCT_METRICS_FORMATTER[metric_name]
        tab_name = STRUCT_METRICS_NAME[metric_name]
        report_tab[tab_name] = formatter.format(val_mean) + '/' + formatter.format(val_median)
    
    return pd.Series(report_tab)


def get_wandb_log(metrics):
    """Get metrics stats for WANDB logging"""

    log_stats = {}
    log_dist = {}
    for metric_name in STRUCT_METRICS_NAME:
        if metric_name not in metrics.columns:
            continue
        
        log_stats[f"{metric_name}_mean"] = metrics[metric_name].mean()
        log_stats[f"{metric_name}_median"] = metrics[metric_name].median()
        log_dist[metric_name] = list(metrics[metric_name].values)

    return log_stats, log_dist

# =============================================================================
# Quanlity functions
# =============================================================================


def compute_secondary_struct(
    pdb_fpath: PATH_TYPE,
):
    """Modified form FramDiff"""

    traj = md.load(str(pdb_fpath))
    pdb_ss = md.compute_dssp(traj, simplified=True)
    pdb_coil_percent = np.mean(pdb_ss == 'C')
    pdb_helix_percent = np.mean(pdb_ss == 'H')
    pdb_strand_percent = np.mean(pdb_ss == 'E')
    pdb_ss_percent = pdb_helix_percent + pdb_strand_percent
    return {
        'ss_percent': pdb_ss_percent,
        'coil_percent': pdb_coil_percent,
        'helix_percent': pdb_helix_percent,
        'strand_percent': pdb_strand_percent, # beta-sheet
    }


def compute_protein_struct_metrics_CA(pdb_fpath: PATH_TYPE) -> Dict[str, float]:
    """Compute protein structural metrics for CA models"""
    res_aa, res_id, atom_list, ca_coords = coords_to_npy(pdb_fpath, atom_list=('CA',))

    return dict(
        CA_clash_rate_framediff=compute_CA_clash_rate(ca_coords, tol=1.9),
        CA_clash_rate_str2str=compute_CA_clash_rate(ca_coords, tol=0.4),
        CA_disconnect_rate_str2str = compute_CA_disconnect_rate(ca_coords),
    )


def get_backbone_bond_distance(
    struct_o_fpath: Union[PATH_TYPE, np.ndarray], model_id: int = 0, chain_id: str = 'first',
    compute_CA_CA: bool = True, compute_C_N: bool = True,   # inter-residue bonds
    compute_N_CA: bool = False, compute_CA_C: bool = False, compute_C_O: bool = False, # intra-residue bonds
    **aa_info_arrays
) -> pd.DataFrame:
    """Compute the protein backbone bond length and CA-CA distance

    Protein backbone struct:
                O            R     O
                |            |     | 
        [... -- C] -- [ N -- CA -- C] -- [N -- ...]
          prev AA         this AA         next AA

    Backbone bonds include:
        Intra-residue bonds: N-CA, CA-C, C=O 
        Inter-residue bonds: C-N

    Returns:
        pd.DataFrame: a table of backbone bond distances and CA distances

    """
    return_val = dict()
    if isinstance(struct_o_fpath, np.ndarray):
        # precomputed backbone coords
        backbone_coords = struct_o_fpath
        seqlen = backbone_coords.shape[0]
        assert backbone_coords.shape == (seqlen, 4, 3), \
             "Input error: not a valid np.ndarray of shape (N, 4, 3)"
    else:
        # PDB file
        res_aa, res_id, atom_list, backbone_coords = coords_to_npy(
            struct_o_fpath=struct_o_fpath, model_id=model_id, chain_id=chain_id,
            atom_list=('N', 'CA', 'C', 'O')
        )
        assert backbone_coords.shape == (len(res_aa), len(atom_list), 3)
        return_val['resid'] = res_id
        return_val['aa'] = res_aa
    return_val.update(aa_info_arrays)
    
    if compute_CA_CA:
        dist_CA = np.linalg.norm(backbone_coords[1:, 1, :] - backbone_coords[:-1, 1, :], axis=1)
        return_val['CA-CA'] = np.concatenate([dist_CA, [np.nan]])
    if compute_C_N:
        dist_CN = np.linalg.norm(backbone_coords[1:, 0, :] - backbone_coords[:-1, 2, :], axis=1)
        return_val['C-N'] = np.concatenate([dist_CN, [np.nan]])
    if compute_N_CA:
        return_val['N-CA'] = np.linalg.norm(backbone_coords[:, 1, :] - backbone_coords[:, 0, :], axis=1)
    if compute_CA_C:
        return_val['CA-C'] = np.linalg.norm(backbone_coords[:, 2, :] - backbone_coords[:, 1, :], axis=1)
    if compute_C_O:
        return_val['C=O'] = np.linalg.norm(backbone_coords[:, 3, :] - backbone_coords[:, 2, :], axis=1)
    
    return pd.DataFrame(return_val)


def get_CA_pairwise_distance(
    struct_o_fpath: Union[PATH_TYPE, np.ndarray], model_id: int = 0, chain_id: str = 'first',
    format_: Literal['matrix', 'flat'] = 'matrix',
):
    """Compute the pairwise CA-CA distance. Unit: angstrom
    """
    if isinstance(struct_o_fpath, np.ndarray):
        # CA coords
        ca_coords = np.squeeze(struct_o_fpath)
        assert len(ca_coords.shape) == 2 and ca_coords.shape[1] == 3, \
            "Input error: not a valid np.ndarray of shape (N, 3)"
        res_aa = None
        res_id = None
    else:
        res_aa, res_id, _, ca_coords = coords_to_npy(
            struct_o_fpath=struct_o_fpath, model_id=model_id, chain_id=chain_id,
            atom_list=('CA',)
        )
        ca_coords = np.squeeze(ca_coords)
        assert ca_coords.shape == (len(res_aa), 3), f"Shape mismatch: {ca_coords.shape} vs {(len(res_aa), 3)}"
    
    if format_ == 'matrix':
        pdist = ca_coords[None, :, :] - ca_coords[:, None, :]
        pdist = np.sqrt((pdist ** 2).sum(axis=-1))
        np.fill_diagonal(pdist, np.nan)
    else:
        assert format_ == 'flat'
        seqlen = ca_coords.shape[0]
        src, dst = list(zip(*combinations(np.arange(seqlen), 2))) # zero-based index
        pdist = ca_coords[src, :] - ca_coords[dst, :]
        pdist = np.sqrt((pdist ** 2).sum(axis=-1))
    return res_aa, res_id, pdist


def compute_CA_clash_rate(ca_coords, C_radius=residue_constants.van_der_waals_radius['C'], tol=1., k=1) -> float:
    """Compute the CA-CA clash rate = num of clashes / num valid distance pairs

    Clash is defined as
        distance < C_radius * 2 - tol
    
        Default tol:
            FrameDiff: overal threshold is 1.5 A, approx tol = 1.9 
            str2str: tol = 0.4
    
    Args:
        C_radius (float): C wdv radius. Should be 1.7 Angstrom.
        tol (float): tolerance for clashing.
        k (int): exclude k neighboring CAs
    """
    _, _, pdist = get_CA_pairwise_distance(ca_coords, format_='matrix')

    row, col = np.triu_indices(pdist.shape[0], k=k)
    pdist = pdist[row, col]
    mask = ~np.isnan(pdist)
    return np.sum(pdist[mask] < C_radius * 2 - tol) / np.sum(mask)


def compute_CA_disconnect_rate(ca_coords, expect=residue_constants.ca_ca, tol=0.4) -> float:
    """Compute the CA-CA disconnection rate

    Connection is broke if
        distance > expect + tol
    
    Default:
        expect: 3.80 Angstrom from AF2 default
        FrameDiff uses tol = 0.1 -> 3.90 max
        We survey the maximum neighboring CA-CA distance from MD trajectories and it is consistent at 4.19 A
            Correspondingly, the tol is 0.4 A
    
    Args:
        expect (float): expected CA-CA distance. Unit: Angstrom. default: 3.80
        tol (float): tolerance for broken connection. Default is 0.4 A from MD struct survey
    """
    dist = np.linalg.norm(ca_coords[:-1, :] - ca_coords[1:, :], axis=-1)
    mask = ~np.isnan(dist)
    return np.mean(dist[mask] > expect + tol)


def compute_bad_CN_bond_rate_biopython(
    bond_dist: Union[np.ndarray, pd.Series],
    max_peptide_length=1.4,
):
    """Compute the rate of bad inter-residue (CN) bond
    Bad bond is defined as greater than maxPeptideBond (1.4 A):
        https://biopython.org/docs/dev/api/Bio.PDB.internal_coords.html#Bio.PDB.internal_coords.IC_Chain
    """
    mask = ~np.isnan(bond_dist)
    return np.mean(bond_dist[mask] > max_peptide_length)


def get_chain_internal_coords_biopython(
    struct_o_fpath, model_id: int = 0, chain_id: str = 'first'
) -> pd.DataFrame:
    """Extract chain internal coordinates using biopython module `atom_to_internal_coordinates`

    It only computes the internal coordinates when satisfy the protein physics (e.g., bond not too long)

    """
    chain = get_protein_chain(struct_o_fpath, model_id=model_id, chain_id=chain_id)
    
    # Use biopython's internal function
    #   np.nan are returned if unpreferred value encountered (e.g., chain break)
    chain.atom_to_internal_coordinates(verbose=False)
    # structure_rebuild_test(chain)
    all_info = []
    for res in chain.get_residues():
        if res.id[0] != ' ':
            # non-protein residue
            continue
        all_info.append(
            {
                'resid': res.get_full_id()[-1][1],
                'aa': res2aacode(res),
                'psi': res.internal_coord.get_angle("psi"),  # N:CA:C:1N
                'phi': res.internal_coord.get_angle("phi"),  # -1C:N:CA:C
                'omg': res.internal_coord.get_angle("omg"),  # -1CA:-1C:N:CA
                'tau': res.internal_coord.get_angle("tau"),  # N:CA:C
                'C:1N:1CA': res.internal_coord.get_angle("C:1N:1CA"), # between residue angle
                'CA:C:1N': res.internal_coord.get_angle("CA:C:1N"), # between residue angle
                'C:1N': res.internal_coord.get_length('C:1N'),
                'N:CA': res.internal_coord.get_length('N:CA'),
                'CA:C': res.internal_coord.get_length('CA:C')
            }
        )
    return pd.DataFrame(all_info)
