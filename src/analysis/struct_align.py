"""Methods for structure alignment

We use third party tools (tmscore and lddt) for structure evaluation. Please install the tools following the instruction:

    - Install TM-score: https://zhanggroup.org/TM-score/
    - Install lddt: https://swissmodel.expasy.org/lddt/downloads/

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Union, Dict, Optional, List

import os
import torch
import subprocess
import numpy as np
import pandas as pd
from pathlib import PosixPath, Path



from openfold.np import residue_constants as rc
from openfold.data import data_transforms
import openfold.utils.loss as af2_loss
from Bio.PDB import PDBParser
import torch

from src.utils.hydra_utils import get_pylogger
from src.utils.misc.process import mp_imap

logger = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================
PATH_TYPE = Union[str, PosixPath]

_TMSCORE = 'tools/TMSCORE'
_LDDT = 'tools/lddt'

pdb_parser = PDBParser(QUIET=True)

METRIC_FORMATTER = {
    'RMSD': '{:.2f}',
    'TMscore': '{:.3f}',
    'GDT-TS': '{:.3f}',
    'lDDT': '{:.3f}',
    'SC-FAPE': '{:.2f}',
}


# =============================================================================
# Align functions
# =============================================================================


def tmscore(
    test_struct: PATH_TYPE,
    ref_struct: PATH_TYPE,
    tmscore_exec=_TMSCORE, 
    verbose=False
):
    """Get alignment scores using TMscore (seq-based alignment)

    NOTE:
        TMscore is normalized by the length of the ref_struct
    
        This function uses `TMscore -seq ...` for alignment calculation. The alignment is based on
            sequence alignment and thus not suitable to compare two different proteins
            For sequence-agnostic alignment and comparison, uses TMalign (https://zhanggroup.org/TM-align/)

    -----------------
    Modified from EigenFold (utils.pdb.tmscore, https://github.com/bjing2016/EigenFold/blob/master/utils/pdb.py)
    
    [LICENSE]
    MIT License

    Copyright (c) 2023 Bowen Jing, Ezra Erives, Peter Pao-Huang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    ------------------
    See TMscore at https://zhanggroup.org/TM-score/
    """

    assert os.path.exists(test_struct), f"test_struct not found: {test_struct}"
    assert os.path.exists(ref_struct), f"ref_struct not found: {ref_struct}"

    if tmscore_exec is None:
        raise ValueError(
            "Please set ENV var TMSCORE or provide tmscore_exec to compute TM scores.\n"
            "See https://zhanggroup.org/TM-score/ for installing TMscore"
        )
    
    out = 'TMscore subprocess failed' # if no 'out' return 
    try:
        # run subprocess TMscore    
        out = subprocess.check_output(
            [str(tmscore_exec), '-seq', test_struct, ref_struct], stderr=open('/dev/null', 'w'), timeout=30
        )
        start = out.find(b'RMSD')
        end = out.find(b'rotation')
        out = out[start:end]

        # parse output
        rmsd, _, tm, _, gdt_ts, gdt_ha, _, _ = out.split(b'\n')
        rmsd = float(rmsd.split(b'=')[-1])
        tm = float(tm.split(b'=')[1].split()[0])
        gdt_ts = float(gdt_ts.split(b'=')[1].split()[0])
        gdt_ha = float(gdt_ha.split(b'=')[1].split()[0])
    except Exception as e:
        rmsd = tm = gdt_ts = gdt_ha = np.nan
        if verbose:
            logger.error(f"TMscore error: {e}\nref: {ref_struct}\ntest: {test_struct}\noutput:\n{out}")

    return {'RMSD': rmsd, 'TMscore': tm, 'GDT-TS': gdt_ts, 'GDT-HA': gdt_ha}


def lddt(
    test_struct: PATH_TYPE,
    ref_struct: PATH_TYPE,
    lddt_exec=_LDDT,
    verbose=True
):
    assert os.path.exists(test_struct), f"test_struct not found: {test_struct}"
    assert os.path.exists(ref_struct), f"ref_struct not found: {ref_struct}"

    lddt = np.nan
    try:
        out = subprocess.check_output(
            [str(lddt_exec), '-xc', str(test_struct), str(ref_struct)],  # reference comes last
            stderr=open('/dev/null', 'w'), timeout=30
        )
        for line in out.split(b'\n'):
            if b'Global LDDT score' in line:
                lddt = float(line.split(b':')[-1].strip())
    except Exception as e:
        if verbose:
            logger.error(f"Error renumbering sequences for {ref_struct}: {e}")
    
    return {'lDDT': lddt}


def sidechain_fape(
    test_struct: PATH_TYPE,
    ref_struct: PATH_TYPE,
    seqres: str
):
    """Compute side-chain FAPE loss between two PDB structures"""
    assert os.path.exists(test_struct), f"test_struct not found: {test_struct}"
    assert os.path.exists(ref_struct), f"ref_struct not found: {ref_struct}"

    def _load_openfold_feat(pdb_path, seqres):
        struct = pdb_parser.get_structure('', pdb_path)
        # chain = struct[0] # each PDB file contains a single conformation, i.e., model 0
        chain = next(iter(struct[0].child_dict.values()))

        # load atomic coordinates
        atom_coords = np.zeros((len(seqres), rc.atom_type_num, 3)) * np.nan # (seqlen, 37, 3)
        for residue in chain:
            seq_idx = residue.id[1] - 1 # zero-based indexing
            for atom in residue:
                atom_coords[seq_idx, rc.atom_order[atom.name]] = atom.coord
        atom_coords -= np.nanmean(atom_coords, axis=(0, 1), keepdims=True)
        all_atom_positions = torch.from_numpy(atom_coords) # (seqlen, 37, 3)
        all_atom_mask = torch.all(~torch.isnan(all_atom_positions), dim=-1) # (seqlen, 37)
        
        all_atom_positions = torch.nan_to_num(all_atom_positions, 0.) # convert NaN to zero
        aatype = torch.LongTensor(
                [rc.restype_order_with_x[res] for res in seqres]
            )
        openfold_feat_dict = {
                'aatype': aatype.long(),
                'all_atom_positions': all_atom_positions.double(),
                'all_atom_mask': all_atom_mask.double(),
            }
            
        openfold_feat_dict = data_transforms.atom37_to_frames(openfold_feat_dict)
        openfold_feat_dict = data_transforms.make_atom14_masks(openfold_feat_dict)
        openfold_feat_dict = data_transforms.make_atom14_positions(openfold_feat_dict)
        openfold_feat_dict = data_transforms.atom37_to_torsion_angles()(openfold_feat_dict)

        return openfold_feat_dict

    test_feat = _load_openfold_feat(test_struct, seqres)
    ref_feat = _load_openfold_feat(ref_struct, seqres)
    gt_batch = {
        'atom14_gt_positions': ref_feat['atom14_gt_positions'],
        'atom14_alt_gt_positions': ref_feat['atom14_alt_gt_positions'],
        'atom14_atom_is_ambiguous': ref_feat['atom14_atom_is_ambiguous'],
        'atom14_gt_exists': ref_feat['atom14_gt_exists'],
        'atom14_alt_gt_exists': ref_feat['atom14_alt_gt_exists'],
        'atom14_atom_exists': ref_feat['atom14_atom_exists'],
    }
    renamed_gt_batch = af2_loss.compute_renamed_ground_truth(
        batch=gt_batch,
        atom14_pred_positions=test_feat['atom14_gt_positions'],
    )
    sc_fape_loss = af2_loss.sidechain_loss(
        sidechain_frames=test_feat['rigidgroups_gt_frames'][None],
        sidechain_atom_pos=test_feat['atom14_gt_positions'][None],
        rigidgroups_gt_frames=ref_feat['rigidgroups_gt_frames'],
        rigidgroups_alt_gt_frames=ref_feat['rigidgroups_alt_gt_frames'],
        rigidgroups_gt_exists=ref_feat['rigidgroups_gt_exists'],
        renamed_atom14_gt_positions=renamed_gt_batch['renamed_atom14_gt_positions'],
        renamed_atom14_gt_exists=renamed_gt_batch['renamed_atom14_gt_exists'],
        alt_naming_is_better=renamed_gt_batch['alt_naming_is_better'],
    )
    return {'SC-FAPE': float(sc_fape_loss)}


# =============================================================================
# Main functions
# =============================================================================

def compute_align_scores(
    test_struct: PATH_TYPE, 
    ref_struct: PATH_TYPE, 
    compute_lddt=False, 
    compute_sc_fape=False, 
    seqres=None, 
    tmscore_exec=_TMSCORE, 
    lddt_exec=_LDDT,
    show_error=True,
) -> Dict[str, float]:
    """Compute alignment scores (TMscore, RMSD, GDT, lddt, etc) between two structures

    Args:
        test_struct (PATH_TYPE): test structure pdb path
        ref_struct (PATH_TYPE): reference structure pdb path
        compute_lddt (bool): if calculate lddt scores
        compute_sc_fape (bool): if compute side-chain FAPE loss
        seqres (str, optional): sequence, required when compute_sc_fape=True
        tmscore_exec (PYTH_TYPE): path to TMscore binary. default path set to ENV variable TMSCORE
        lddt_exec (PYTH_TYPE): path to lddt binary. default path set to ENV variable LDDT
        show_error (bool): if suppress error if encountered
        
    Returns:
        dict of scores
    
    ---
    Install TM-score: https://zhanggroup.org/TM-score/
    Install lddt: https://swissmodel.expasy.org/lddt/downloads/
    """
    
    assert os.path.exists(ref_struct), f"ref_struct not found: {ref_struct}"
    assert os.path.exists(test_struct), f"test_struct not found: {test_struct}"

    align_output = tmscore(
        ref_struct=ref_struct, test_struct=test_struct, 
        tmscore_exec=tmscore_exec, 
        verbose=show_error
    )
    
    if compute_lddt:
        align_output.update(
            lddt(
                test_struct=test_struct, 
                ref_struct=ref_struct, 
                lddt_exec=lddt_exec, 
                verbose=show_error
            )
        )
        
    if compute_sc_fape:
        assert seqres is not None, "seqres is required for side-chain FAPE calculation"
        align_output.update(sidechain_fape(test_struct=test_struct, ref_struct=ref_struct, seqres=seqres))

    return align_output


# =============================================================================
#  Additional alignment tools
# =============================================================================


def _worker_min_rmsd_to_group(fpath, ref_fpath_list):
    """Compute minimum RMSD of fpath to a group of reference structures"""
    results = pd.DataFrame(
        [compute_align_scores(ref_struct=ref_fpath, test_struct=fpath) for ref_fpath in ref_fpath_list]
    )
    results['fname'] = Path(fpath).name
    return results.sort_values('RMSD', ascending=False).iloc[0]


def min_rmsd_to_group(fpath_list, ref_fpath_list, n_proc=None, **kwargs):
    """Align PDB in fpath_list to a list of references and report the minimal RMSD"""
    
    all_results = mp_imap(
        func=_worker_min_rmsd_to_group, iter=fpath_list, n_proc=n_proc, mute_tqdm=True,
        ref_fpath_list=ref_fpath_list, **kwargs
    )
    
    return pd.DataFrame(all_results)


def _worker_align_to_group(fpath, ref_fpath_list, **align_kwargs):
   
    results = pd.Series({
        f'{key}_{ref_fpath.stem}': val for ref_fpath in ref_fpath_list for key, val in compute_align_scores(
            ref_struct=ref_fpath, 
            test_struct=fpath, 
            **align_kwargs
        ).items()
    })
    results['min_RMSD'] = results[[f'RMSD_{ref_fpath.stem}' for ref_fpath in ref_fpath_list]].min()
    results['max_TMscore'] = results[[f'TMscore_{ref_fpath.stem}' for ref_fpath in ref_fpath_list]].max()
    results['fname'] = Path(fpath).name
    return results


def align_to_group(fpath_list, ref_fpath_list, n_proc=None, **kwargs):
    """Align fpath to a group of reference structures. Report all distances."""
    all_results = mp_imap(
        func=_worker_align_to_group, iter=fpath_list, n_proc=n_proc,
        ref_fpath_list=ref_fpath_list, **kwargs
    )
    return pd.DataFrame(all_results)
