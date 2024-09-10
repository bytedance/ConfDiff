"""Use FASPR to add side chain atoms for protein backbones (side-chain packing)

See https://github.com/tommyhuangthu/FASPR to install

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from time import perf_counter
from argparse import ArgumentParser

from src.utils.misc.process import subprocess_run, mp_imap_unordered
from src.utils.misc.misc import get_persist_tmp_fpath
from src.utils import hydra_utils

logger = hydra_utils.get_pylogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def faspr_pack(input_fpath, output_fpath, faspr_exec=None):
    """Run FASPR packing"""
    if output_fpath is None:
        output_fpath = get_persist_tmp_fpath()
    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    out, err = subprocess_run(
        cmd = [faspr_exec, '-i', str(input_fpath), '-o', str(output_fpath)],
        quiet=True
    )
    
    if err != '':
        logger.error(f"FASPR error: {err}")
        return None
    return output_fpath


def _work_fn(input_output, **kwargs):
    input_fpath, output_fpath = input_output
    return faspr_pack(input_fpath, output_fpath, **kwargs)


def pack_all_pdbs(input_root, output_root, n_proc=10, **kwargs):
    """Run side-chain packing for all proteins under input_root and save packed structure to output_root.
    
    All pdb files will be find recursively and the output_root will keep the same directory
    """
    input_root = Path(input_root)
    output_root= Path(output_root)

    input_output_list = [
        (in_fpath, output_root.joinpath(str(in_fpath.relative_to(input_root)).replace('.pdb', '') + '_faspr_packed.pdb'))
        for in_fpath in input_root.rglob('*.pdb')
    ]

    start_t = perf_counter()
    results = mp_imap_unordered(
        func=_work_fn, iter=input_output_list, n_proc=n_proc,
        **kwargs
    )
    end_t = perf_counter()

    failed = sum([res is None for res in results])

    print(f">>> FASPR packed {len(input_output_list)} pdb in {end_t - start_t:.1f} sec. {failed} Failed.")

# =============================================================================
# Classes
# =============================================================================

if __name__ == "__main__":
    
    parser = ArgumentParser(prog="FASPR side-chain packing")
    parser.add_argument('--input-root', type=str, required=True)
    parser.add_argument('--output-root', type=str, required=True)
    parser.add_argument('--n-proc', type=int, default=10)
    args = parser.parse_args()

    pack_all_pdbs(args.input_root, args.output_root, n_proc=args.n_proc)
