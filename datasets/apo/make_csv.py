"""Make csv file for apo-holo: ligand-induced protein conformation change benchmark

Source: Impact of protein conformational diversity on AlphaFold predictions (Saldaño et al, 2022, Bioinformatics)

It contains 91 curated pairs of conformation changes involved in ligand-binding

List of PDB pairs can be obtained from one of the sources:

- Supplementary data at https://academic.oup.com/bioinformatics/article/38/10/2742/6563595?login=false: Supplementary_Table_1_91_apo_holo_pairs.csv
- Original gitlab: https://gitlab.com/sbgunq/publications/af2confdiv-oct2021/-/blob/main/data/26-01-2022/revision1_86_plus_5.csv

Special cases as in EigenFold:
  - 2BNH_0_A contains unknown amino acid ('X') in seqres and is excluded in EigenFold's 90 cases.

----------------
[License]

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
import os
import argparse
import pandas as pd

from Bio.PDB import PDBIO

from ..rcsb import mmcif_parsing
from ..rcsb.utils import prune_chain, AtomSelect

from src.analysis.struct_align import compute_align_scores
from src.utils.misc.process import mp_imap

# =============================================================================
# Functions
# =============================================================================

def process_row(row):
    # apo info
    apo_pdb_id, apo_chain_id = row.apo_id.split('_')
    if len(apo_pdb_id) > 4:
        assert '-' in apo_pdb_id
        apo_pdb_id, apo_pdb_model_id = apo_pdb_id.split('-')
        apo_model_id = int(apo_pdb_model_id) - 1 # zero-based model ID
    else:
        apo_model_id = 0 # zero-based model ID
    # holo info
    holo_pdb_id, holo_chain_id = row.holo_id.split('_')
    if len(holo_pdb_id) > 4:
        assert '-' in holo_pdb_id
        holo_pdb_id, holo_pdb_model_id = holo_pdb_id.split('-')
        holo_model_id = int(holo_pdb_model_id) - 1 # zero-based model ID
    else:
        holo_model_id = 0 # 1-based PDB model index# zero-based model ID

    # parse apo mmcif
    apo_pid = apo_pdb_id.lower()
    apo_mmcif_path = os.path.join(args.mmcif_dir, apo_pid[1:3], f'{apo_pid}.cif.gz')
    apo_mmcif_object = mmcif_parsing.parse(apo_mmcif_path)
    apo_header = apo_mmcif_object.header
    apo_structure = apo_mmcif_object.full_structure
    apo_chain_to_seqres = apo_mmcif_object.chain_to_seqres
    apo_struct_mappings = apo_mmcif_object.struct_mappings
    # apo metadata
    apo_chain = apo_structure[apo_model_id][apo_chain_id]
    apo_chain_map = apo_struct_mappings[apo_model_id][apo_chain_id]
    apo_structure_method = apo_header['structure_method']
    apo_resolution = apo_header['resolution']
    apo_seqres = apo_chain_to_seqres[apo_chain_id] 

    # parse holo mmcif
    holo_pid = holo_pdb_id.lower()
    holo_mmcif_path = os.path.join(args.mmcif_dir, holo_pid[1:3], f'{holo_pid}.cif.gz')
    holo_mmcif_object = mmcif_parsing.parse(holo_mmcif_path)
    holo_header = holo_mmcif_object.header
    holo_structure = holo_mmcif_object.full_structure
    holo_chain_to_seqres = holo_mmcif_object.chain_to_seqres
    holo_struct_mappings = holo_mmcif_object.struct_mappings
    # holo metadata
    holo_chain = holo_structure[holo_model_id][holo_chain_id]
    holo_chain_map = holo_struct_mappings[holo_model_id][holo_chain_id]
    holo_structure_method = holo_header['structure_method']
    holo_resolution = holo_header['resolution']
    holo_seqres = holo_chain_to_seqres[holo_chain_id] 

    # save apo PDB file
    apo_pdb_path = os.path.join(
        args.output_pdb_dir,
        apo_pdb_id[1:3],
        apo_pdb_id,
        f'{apo_pdb_id}_{apo_model_id}_{apo_chain_id}.pdb'
    )
    if not os.path.exists(apo_pdb_path): 
        # print(f'Saving {apo_pdb_id} PDB file to {apo_pdb_path}.')
        os.makedirs(os.path.join(args.output_pdb_dir, apo_pdb_id[1:3], apo_pdb_id), exist_ok=True)
        # Biopython chain -> PDB file
        apo_chain_pruned = prune_chain(
            chain=apo_chain,
            chain_map=apo_chain_map
        )
        # save as PDB file
        pdbio = PDBIO()
        pdbio.set_structure(apo_chain_pruned)
        pdbio.save(apo_pdb_path, select=AtomSelect())
    
    # save holo PDB file
    holo_pdb_path = os.path.join(
        args.output_pdb_dir,
        holo_pdb_id[1:3],
        holo_pdb_id,
        f'{holo_pdb_id}_{holo_model_id}_{holo_chain_id}.pdb'
    )
    if not os.path.exists(holo_pdb_path): 
        # print(f'Saving {holo_pdb_id} PDB file to {holo_pdb_path}.')
        os.makedirs(os.path.join(args.output_pdb_dir, holo_pdb_id[1:3], holo_pdb_id), exist_ok=True)
        # Biopython chain -> PDB file
        holo_chain_pruned = prune_chain(
            chain=holo_chain,
            chain_map=holo_chain_map
        )
        # save as PDB file
        pdbio = PDBIO()
        pdbio.set_structure(holo_chain_pruned)
        pdbio.save(holo_pdb_path, select=AtomSelect())
    
    # compute TMscore
    alignment_results = compute_align_scores(
        ref_struct=apo_pdb_path,
        test_struct=holo_pdb_path,
        compute_lddt=False, 
        compute_sc_fape=False,
        tmscore_exec=args.tmscore_exec,
    )
    tmscore = alignment_results['TMscore']
    tmscore_rmsd = alignment_results['RMSD']
    gdt_ts = alignment_results['GDT-TS']
    gdt_ha = alignment_results['GDT-HA']

    if abs(tmscore_rmsd - row.rmsd_apo_holo) > 0.1:
        print(f'{apo_pdb_id}/{holo_pdb_id} large deviation between computed ' + \
            f'and original RMSD: {tmscore_rmsd:.2f} vs {row.rmsd_apo_holo:.2f}')

    return {
        'chain_name': f'{apo_pid}_{apo_model_id}_{apo_chain_id}',
        'seqres': apo_seqres,
        'seqlen': len(apo_seqres),
        # pair info
        'tmscore': tmscore,
        'tmscore_rmsd': tmscore_rmsd,
        'gdt_ts': gdt_ts,
        'gdt_ha': gdt_ha,
        # apo info
        'apo_chain_name': f'{apo_pid}_{apo_model_id}_{apo_chain_id}',
        'apo_pdb_id': apo_pid,
        'apo_model_id': apo_model_id,
        'apo_chain_id': apo_chain_id,
        'apo_structure_method': apo_structure_method,
        'apo_resolution': apo_resolution,
        'apo_seqres': apo_seqres,
        'apo_seqlen': len(apo_seqres),
        # holo info
        'holo_chain_name': f'{holo_pid}_{holo_model_id}_{holo_chain_id}',
        'holo_pdb_id': holo_pid,
        'holo_model_id': holo_model_id,
        'holo_chain_id': holo_chain_id,
        'holo_structure_method': holo_structure_method,
        'holo_resolution': holo_resolution,
        'holo_seqres': holo_seqres,
        'holo_seqlen': len(holo_seqres),
    }

# =============================================================================
# Main
# =============================================================================


def main():
    df = pd.read_csv(args.apo_orig_csv_path, index_col=None, sep=';')
    row_list = [row for _, row in df.iterrows()]

    apo_df = mp_imap(
        func=process_row, iter=row_list,
        n_proc=args.num_workers,
    )

    apo_df = pd.DataFrame(apo_df)
    apo_df.to_csv(args.output_csv_path, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Make apo/holo csv file.')
    parser.add_argument(
        '--apo_orig_csv_path',
        help='Path to the original apo csv file.',
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
        '--tmscore_exec',
        help='Path to TMscore executable',
        default=os.environ.get('TMSCORE', None)
    )
    parser.add_argument(
        '--num_workers',
        help='Number of parallel workers',
        default=1,
        type=int
    )
    
    args = parser.parse_args()


    main()
