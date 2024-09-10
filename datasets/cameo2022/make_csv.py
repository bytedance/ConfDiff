"""Make csv file for CAMEO2022 single structure predition benchmark

- cameo2022_original.csv: original csv file of CAMEO targets between 2022-01-07 to 2022-12-31. 
  It is downloaded from https://www.cameo3d.org/modeling/targets/1-year/?to_date=2022-12-31.

- Special cases in EigenFold's dataset
    - 7OCJ_0_G was excluded in EigenFold: it does not have .pdb file due to large size and not included in EigenFold's data. We include this case.
    - 8ahp_A is superseded by 8qcw_A in PDB.
    

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
import os
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path, PosixPath

from Bio.PDB import PDBIO
from ..rcsb import mmcif_parsing
from ..rcsb.utils import prune_chain, AtomSelect
from src.utils.misc.process import mp_imap

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================
def save_pdb(mmcif_object, pdb_id, chain_id, pdb_fpath, model_id=0):
    """Save single-chain PDB from mmcif"""
    full_structure = mmcif_object.full_structure
    chain = full_structure.child_dict[model_id].child_dict[chain_id]
    chain_map = mmcif_object.struct_mappings[0][chain_id]

    try:
        # Biopython chain -> PDB file
        chain_pruned = prune_chain(
            chain=chain,
            chain_map=chain_map
        )
        # save as PDB file
        pdbio = PDBIO()
        pdbio.set_structure(chain_pruned)
        pdb_fpath.parent.mkdir(exist_ok=True, parents=True)
        pdbio.save(str(pdb_fpath), select=AtomSelect())
    except Exception as e:
        # skip chains that could not be saved by PDBIO
        Path(pdb_fpath).unlink(missing_ok=True)
        print(f"PDB save error: {pdb_id}_{model_id}_{chain_id}: {e}")


def process_row(row):
    pdb_id = row['ref. PDB [Chain]'].split(' ')[0].upper() 
    assert len(pdb_id) == 4, f"Error parsing PDB ID: {pdb_id}"
    chain_id = row['ref. PDB [Chain]'][row['ref. PDB [Chain]'].find('[')+1:row['ref. PDB [Chain]'].rfind(']')].upper()
    if args.output_pdb_dir is not None:
        pdb_dir = Path(args.output_pdb_dir)
    else:
        pdb_dir = None

    # Special case
    if pdb_id == '8AHP':
        # 8AHP_A is superseded by 8QCW_A
        pdb_id, chain_id = '8QCW', 'A'

    # parse mmcif
    pid = pdb_id.lower()
    mmcif_path = os.path.join(args.mmcif_dir, pid[1:3], f'{pid}.cif.gz')
    if not os.path.exists(mmcif_path):
        print(f"MMCIF not found: {mmcif_path}. Skip")
        return None

    mmcif_object = mmcif_parsing.parse(mmcif_path)
    header = mmcif_object.header
    structure_method = header['structure_method']
    resolution = header['resolution']
    release_date = header['release_date']
    chain_to_seqres = mmcif_object.chain_to_seqres
    seqres = chain_to_seqres[chain_id]
    
    # filter as specified in EigenFold (Jin, 2023). It should leave 183 proteins.
    if pid != '8qcw' and (
        len(seqres) >= 750 or \
        release_date >= '2022-11-01' or \
        release_date < '2022-08-01'
    ):
        return None
    
    if pdb_dir is not None:
        save_pdb(
            mmcif_object=mmcif_object,
            pdb_id=pdb_id, chain_id=chain_id,
            pdb_fpath=pdb_dir/pdb_id[1:3]/pdb_id/f"{pdb_id}_{0}_{chain_id}.pdb"
        )
    
    # metadata
    return {
        'chain_name': f'{pdb_id.lower()}_{chain_id}',
        'pdb_id': pdb_id,
        'chain_id': chain_id,
        'structure_method': structure_method,
        'resolution': resolution,
        "release_date": release_date,
        'seqres': seqres,
        'seqlen': len(seqres),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    cameo2022_metadata = []
    df = pd.read_csv(args.cameo2022_orig_csv_path, index_col=None)
    row_list = [row for _, row in df.iterrows()]

    cameo2022_metadata = mp_imap(
        func=process_row, iter=row_list,
        n_proc=args.num_workers,
    )
    cameo2022_metadata = [info for info in cameo2022_metadata if info is not None]
    print(f"{len(cameo2022_metadata)} record passed.")

    df = pd.DataFrame(cameo2022_metadata)
    df.to_csv(args.output_csv_path, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Make cameo2022 csv file.')
    parser.add_argument(
        '--cameo2022_orig_csv_path',
        help='Path to the original cameo2022 csv file.',
        required=True,
        type=str)
    parser.add_argument(
        '--mmcif_dir',
        help='Path to directory with mmcif files.',
        type=str,
        required=True)
    parser.add_argument(
        '--output_csv_path',
        help='Path to output csv file.',
        type=str,
        required=True)
    parser.add_argument(
        '--output_pdb_dir',
        help='Path to PDB file dir.',
        type=str,
        default=None)
    parser.add_argument(
        '--num_workers',
        help='Number of parallel workers',
        default=1,
        type=int
    )
    args = parser.parse_args()

    main()
