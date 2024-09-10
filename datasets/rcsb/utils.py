from typing import Dict, Mapping, Any, Optional, Literal, List
import copy
import numpy as np

from Bio.PDB import Select, Chain
from openfold.np import residue_constants

from . import errors


def chain_to_npy(
    seqres: str, # mmCIF-parsed seqres
    chain: Chain.Chain, # Biopython-parsed chain
    chain_map: Mapping[int, Any], # mmCIF-parsed chain map
) -> Dict[str, np.ndarray]:
    """
    Convert from Biopython chain to all-atom numpy array.
    """
    seqlen = len(seqres)
    assert seqlen == len(chain_map), 'seqlen != len(chain_map)'

    atom_coords = np.zeros((seqlen, residue_constants.atom_type_num, 3)) * np.nan # (seqlen, 37, 3)
    for seq_idx, res_at_pos in chain_map.items():
        if res_at_pos.is_missing: continue
        residue_id = (' ', res_at_pos.resseq, ' ')
        try:
            residue = chain[residue_id]
        except Exception as e:
            raise errors.ResidueError(f'Failed to locate residue from chain, {e}')
        assert residue.resname == res_at_pos.name, 'Residue name mismatch.'
        assert residue_constants.restype_3to1[residue.resname] == seqres[seq_idx]
        assert residue.id[0] == ' ', 'Non-empty het_flag.'
        assert residue.id[2] == ' ', 'Non-empty insertion code.'
        if not (
            'N' in residue.child_dict and \
            'CA' in residue.child_dict and \
            'C' in residue.child_dict and \
            'O' in residue.child_dict
        ):
            continue
        for atom in residue:
            if (
                atom.is_disordered() or \
                (atom.get_altloc() != ' ') or \
                (atom.get_occupancy() < 0.5) or \
                (atom.name not in residue_constants.atom_types)
            ):
                continue
            assert np.all(np.isnan(atom_coords[seq_idx, residue_constants.atom_order[atom.name]])), \
                'Duplicate atom_coords value assignment.'
            assert np.all(~np.isnan(atom.coord)), 'NaN in atom.coord'
            atom_coords[seq_idx, residue_constants.atom_order[atom.name]] = atom.coord
    
    return {
        'atom_coords': atom_coords,
    }


def prune_chain(
    chain: Chain.Chain, # Biopython-parsed chain
    chain_map: Mapping[int, Any], # mmCIF-parsed chain map
) -> Chain.Chain:
    """
    Remove hetero and low-quality residues, renumber residue indices to start from 1.
    """
    chain_copy = copy.deepcopy(chain)
    
    # map from Biopython residue sequence index to seq_idx in seqres
    resseq_to_seq_idx_mapping = {} 
    for seq_idx, res_at_pos in chain_map.items():
        if res_at_pos.is_missing: continue
        assert res_at_pos.resseq not in resseq_to_seq_idx_mapping, \
            'Duplicate res_at_pos.resseq assignment.'
        resseq_to_seq_idx_mapping[res_at_pos.resseq] = seq_idx
    
    # remove HETATM, non-standard/missing/low-quality residues
    for resi in list(chain_copy):
        if (
            (resi.resname not in residue_constants.restype_3to1) or \
            (resi.id[0] != ' ') or (resi.id[2] != ' ') or \
            (resi.id[1] not in resseq_to_seq_idx_mapping) or \
            ('N' not in resi.child_dict) or \
            ('CA' not in resi.child_dict) or \
            ('C' not in resi.child_dict) or \
            ('O' not in resi.child_dict)
        ):   
            chain_copy.detach_child(resi.id)
    
    # renumber PDB residues
    for resi in chain_copy:
        resi.id = (resi.id[0], 10000+resi.id[1], resi.id[2])
    for resi in chain_copy:
        seq_idx = resseq_to_seq_idx_mapping[resi.id[1]-10000]
        resi.id = (resi.id[0], seq_idx+1, resi.id[2]) # 1-based residue index, consistent with seqres
    
    return chain_copy


class AtomSelect(Select):
    def accept_atom(self, atom):
        accept_criteria = not (
            atom.is_disordered() or \
            (atom.get_altloc() != ' ') or \
            (atom.get_occupancy() < 0.5) or \
            (atom.name not in residue_constants.atom_types)
        )
        return accept_criteria
