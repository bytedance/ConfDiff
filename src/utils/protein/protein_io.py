"""IO utilities for proteins files

----------------
[License]
SPDX-License-Identifier: Apache-2.0
---------------------
Copyright (2024) Bytedance Ltd. and/or its affiliates

OR

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.
"""

# =============================================================================
# Imports
# =============================================================================

from typing import Union, Optional, Tuple, Dict, List
import tempfile
import os
import warnings
import numpy as np
from pathlib import PosixPath, Path

from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import Select, PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain

from ..hydra_utils.pylogger import get_pylogger
from .seq import pairwise_globalxx, res2aacode
from src.utils.protein.protein_residues import normal as RESIDUES

logger = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================

pdb_parser = PDBParser(PERMISSIVE=True, QUIET=True)


# =============================================================================
# Functions
# =============================================================================


def load_pdb(fpath, name: Optional[str] = None) -> Structure:
    """Load Bio.Struture from .pdb/.ent/.pdb.gz files"""
    fpath = str(fpath)
    if name is None:
        name = Path(fpath).name
    parser = PDBParser(QUIET=True)
    if fpath.endswith(".gz"):
        import gzip
        with gzip.open(fpath, "rt") as handle:
            struct = parser.get_structure(name, handle)
    else:
        struct = parser.get_structure(name, fpath)
    return struct

def write_pdb(struct, fpath):
    """Save a Bio.Structure object to pdb file"""
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(fpath))


def split_pdb_models(pdb_fpath, output_dir=None):
    """Split models in pdb_fpath and save under to output_dir. If output_dir is None, a temporary dir is provided

    NOTE: remember to clean the tempdir upon finishing.
    """
    struct = load_pdb(pdb_fpath)
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for model_ix, model in enumerate(struct.get_models()):
        write_pdb(model, output_dir / f'{Path(pdb_fpath).stem}_sample{model_ix}.pdb')
    return output_dir


def get_protein_chain(
    struct_o_fpath,
    model_id: int = 0,
    chain_id: str = "first",
) -> Chain:
    """Extract the Bio.Chain object from multiple input format"""
    struct = struct_o_fpath
    if isinstance(struct, (str, PosixPath)):
        struct = pdb_parser.get_structure("", struct)
    if isinstance(struct, Structure):
        struct = struct[model_id]
    if isinstance(struct, Model):
        struct = struct.child_list[0] if chain_id == "first" else struct[chain_id]
    return struct


def coords_to_npy(
    struct_o_fpath,
    model_id: int = 0,
    chain_id: str = "first",
    atom_list: Tuple = ("N", "CA", "C", "O"),
    unk_aa_symbol="X",
) -> Tuple[str, np.ndarray, Tuple, np.ndarray]:
    """Convert atom coords to npy format"""

    chain = get_protein_chain(
        struct_o_fpath=struct_o_fpath, model_id=model_id, chain_id=chain_id
    )

    # get residue index info from PDB
    all_residue_ids = [
        res.get_id()[1] for res in chain.get_residues() if res.id[0] == " "
    ]
    base_idx = min(all_residue_ids)
    max_idx = max(all_residue_ids)
    seqlen = max_idx - base_idx + 1
    num_atoms = len(atom_list)

    res_id = np.ones((seqlen)) * -1
    res_aa = [unk_aa_symbol] * seqlen  # unknown residue symbol
    all_coords = np.zeros((seqlen, num_atoms, 3)) * np.nan

    for res in chain.get_residues():
        if res.id[0] != " ":
            # non-protein residue
            continue
        res_idx = res.get_id()[1]
        res_idx_0base = res_idx - base_idx

        res_id[res_idx_0base] = res_idx
        res_aa[res_idx_0base] = res2aacode(res.get_resname())
        atom_coords = {a.name: a.coord for a in res.get_atoms()}
        all_coords[res_idx_0base] = np.array(
            [atom_coords.get(atom, [np.nan, np.nan, np.nan]) for atom in atom_list]
        )
    res_aa = "".join(res_aa)
    return res_aa, np.array(res_id, dtype=np.int16), atom_list, all_coords

