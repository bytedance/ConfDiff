"""Unpack protein structures from trajectory files (.xtc) as separate .pdb files

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

# =============================================================================
# Imports
# =============================================================================

import os
from pathlib import Path
from argparse import ArgumentParser

import mdtraj
import MDAnalysis as mda
from src.utils.misc.process import mp_imap_unordered

# =============================================================================
# Constants
# =============================================================================

CPU_COUNT = os.cpu_count()


# =============================================================================
# Functions
# =============================================================================


def unpack_protein(chain_name, output_root, traj_root, heavy_only=True):
    output_root.joinpath(chain_name).mkdir(parents=True, exist_ok=True)
    
    for rep in [1, 2, 3]:
        xtc_fpath = traj_root/chain_name/f"{chain_name}_prod_R{rep}_fit.xtc"
        top_fpath = traj_root/chain_name/f"{chain_name}.pdb"
        traj = mdtraj.load(str(xtc_fpath), top=str(top_fpath))
        if heavy_only:
            traj = traj.atom_slice(traj.topology.select_atom_indices('heavy'))
        for frame_idx in range(traj.n_frames):
            traj[frame_idx].save_pdb(str(output_root/chain_name/f"{chain_name}_prod_R{rep}_frame{frame_idx}.pdb"))


def unpack_all_protein(traj_root, output_root, heavy_only=True, n_proc=None):
    traj_root = Path(traj_root)
    output_root = Path(output_root)

    chain_name_list = [subdir.stem for subdir in traj_root.glob('*') if subdir.is_dir()]
    print(f"Unpacking pdb from {len(chain_name_list)} proteins. Save to {output_root}")

    mp_imap_unordered(
        func=unpack_protein, iter=chain_name_list, n_proc=n_proc,
        traj_root=traj_root, output_root=output_root, heavy_only=heavy_only
    )


# =============================================================================
# Classes
# =============================================================================

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--traj-root', type=str)
    parser.add_argument('--output-root', type=str)
    parser.add_argument('--n-proc', type=int, default=None)
    parser.add_argument('--heavy-only', type=lambda x: eval(x), default=True)
    args = parser.parse_args()

    unpack_all_protein(
        traj_root=args.traj_root,
        output_root=args.output_root,
        heavy_only=args.heavy_only,
        n_proc=args.n_proc,
    )
