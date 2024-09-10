"""Analysis tools using mdtraj package

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Literal, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

import pyemma
import mdtraj
from mdtraj import Trajectory

# =============================================================================
# Constants
# =============================================================================


# MDTraj filtering criteria
FILTER_HEAVY = 'element != H'
FILTER_HEAVY_NON_TERM = 'name != OXT and element != H'
FILTER_CA = 'name CA'
FILTER_BACKBONE = 'backbone'


# =============================================================================
# Functions
# =============================================================================

def get_atom_index(traj: Trajectory, select: Literal['CA', 'backbone', 'heavy'] = 'CA') -> np.ndarray:
    """Return the selected atom index in the traj"""
    if select == 'CA':
        atom_idx = traj.topology.select(FILTER_CA)
    elif select == 'backbone':
        atom_idx = traj.topology.select(FILTER_BACKBONE)
    elif select == 'heavy':
        atom_idx = traj.topology.select(FILTER_HEAVY)
    else:
        raise ValueError(f'Unknown select: {select}')
    return atom_idx


def get_atom_idx_mapping(target: Trajectory, ref: Trajectory) -> pd.Series:
    """Get atom index mapping from target to ref by atom name

    That is, ref' atom index in target's order

    NOTE:
        All atoms in target MUST exists in ref.
    
    Returns:
        pd.Series: atom index mapping from target atoms to ref atoms
    """
    target_atom_names = [str(atom) for atom in target.topology.atoms]
    ref_atom_names = [str(atom) for atom in ref.topology.atoms]
    ref_atom_name_to_index = pd.Series(np.arange(len(ref_atom_names)), index=ref_atom_names)
    ref_matched_index = ref_atom_name_to_index[target_atom_names]
    assert not ref_matched_index.isna().any(), f'missing atoms in ref structure: {ref_matched_index[ref_matched_index.isna()].index}'
    return ref_matched_index


def safe_superimpose(target: Trajectory, ref: Trajectory, frame: int = 0) -> Trajectory:
    """Safely superimpose two structures by matching atom name. Atoms are matched by atom name ({resid}-{atom name})

    NOTE: 
        all atoms in the target trajectories are used. Filter trajectory if only want to match a subset of atoms.
    """
    ref_matched_index = get_atom_idx_mapping(target, ref)
    ref_matched_index = ref_matched_index.values
    return target.superpose(ref, frame=frame, atom_indices=np.arange(target.n_atoms), ref_atom_indices=ref_matched_index)


def get_pairwise_rmsd(traj, select: Literal['CA', 'backbone'] = 'CA', unit: Literal['A', 'nm'] = 'A') -> np.ndarray:
    """Compute the pairwise RMSD between frames"""
    atom_idx = get_atom_index(traj, select)    
    all_rmsd = []
    for frame_idx in range(traj.n_frames):
        rmsd_to_ref = mdtraj.rmsd(traj, traj, frame=frame_idx, atom_indices=atom_idx)
        all_rmsd.append(rmsd_to_ref)
    all_rmsd = np.array(all_rmsd)

    if unit == 'A':
        all_rmsd *= 10
    elif unit == 'nm':
        pass
    else:
        raise ValueError(f'Unknown unit: {unit}')

    return all_rmsd


def get_rmsf(
    traj: Trajectory, 
    align_to: Optional[Trajectory] = None,
    select: Literal['CA', 'backbone', 'heavy'] = 'CA',
    unit: Literal['A', 'nm'] = 'A',
    skip_align=False
) -> pd.Series:
    """Get the RMSF for each selected atoms"""

    if align_to is None:
        ref = traj = traj.atom_slice(get_atom_index(traj, select))
    else:
        traj = traj.atom_slice(get_atom_index(traj, select))
        ref = align_to.atom_slice(get_atom_index(align_to, select))

    if not skip_align:
        # NOTE: superimpose is required prior to compute rmsf. skip_align = True ONLY IF the traj has been aligned.
        traj = safe_superimpose(traj, ref, frame=0)
    rmsf = mdtraj.rmsf(traj, None)
    atom_names = [str(atom) for atom in traj.topology.atoms]

    if unit == 'A':
        rmsf *= 10
    elif unit == 'nm':
        pass
    else:
        raise ValueError(f'Unknown unit: {unit}')

    return pd.Series(rmsf, index=atom_names)


def get_CA_pairwise_dist(traj: Trajectory, excluded_neighbors=0):
    """Use pyEMMA to get the pairwise distance between CA atoms"""
    CA_atom_idx = traj.top.select('name CA')
    traj_CA = traj.atom_slice(CA_atom_idx, inplace=False)

    # TODO: consider replace pyemma
    featurizer = pyemma.coordinates.featurizer(traj_CA.topology)
    featurizer.add_distances_ca(excluded_neighbors=excluded_neighbors)
    pair_idx = featurizer.active_features[0].distance_indexes
    pdist = featurizer.transform(traj)
    return pdist, pair_idx


def filter_CA_pairwise_dist(pdist, pair_idx, excluded_neighbors=3):
    """Filter out the CA pairs that are too close to each other"""
    offsets = np.abs(pair_idx[:, 0] - pair_idx[:, 1])
    keep = offsets >= excluded_neighbors
    return pdist[:, keep], pair_idx[keep]


def get_radius_of_gyration(traj: Trajectory):
    """Compute the radius of gyration for each atom"""
    CA_atom_idx = traj.top.select('name CA')
    traj_CA = traj.atom_slice(CA_atom_idx, inplace=False)

    com = mdtraj.compute_center_of_mass(traj_CA)
    xyz_centered = traj_CA.xyz - com[:, None, :] # type: ignore
    rog = np.sqrt(np.sum(xyz_centered ** 2, axis=-1))
    return rog


def get_backbone_dihedral_sincos(traj: Trajectory) -> Tuple[np.ndarray, list]:
    """Use pyEMMA to extract the sine and cosine of the backbone dihedral angles (phi, psi)"""
    backbone_atoms = traj.topology.select("protein and backbone and (not resname NME) and (not resname ACE)") # backbone atoms
    traj = traj.atom_slice(backbone_atoms)
    featurizer = pyemma.coordinates.featurizer(traj.topology)
    featurizer.add_backbone_torsions(cossin=True, periodic=True)
    feat = featurizer.transform(traj)
    feat_name = featurizer.describe()
    return feat, feat_name


def _dihedral_feat_name_to_code(feat_name, offset):
    """Convert diheral feature name to unique hashable code"""
    if isinstance(feat_name, (list, np.ndarray)):
        return [_dihedral_feat_name_to_code(feat, offset) for feat in feat_name]
    else:
        parts = feat_name.split(' ')
        assert len(parts) == 4
        tri_fn, angle_type = parts[0].split('(')
        res_idx = int(parts[-1].replace(')', '')) - offset
        return (res_idx, tri_fn, angle_type) # (res_idx, SIN/COS, PHI/PSI)


def align_dihedral_feats(feat_name, ref_feat_name, resid_offset) -> np.ndarray:
    """Get the aligned order of current feat_name to ref_feat_name used for TICA fitting"""
    feat_name = _dihedral_feat_name_to_code(feat_name, resid_offset)
    ref_feat_name = _dihedral_feat_name_to_code(ref_feat_name, offset=0)
    # ensure one-to-one mapping
    assert len(set(feat_name)) == len(set(ref_feat_name))
    assert all(feat in ref_feat_name for feat in feat_name)

    old_idx_mapper = {feat: idx for idx, feat in enumerate(feat_name)}
    new_idx = np.array([old_idx_mapper[feat] for feat in ref_feat_name])
    return new_idx


def filter_termini_dihedrals(feat_list, k=1):
    """Remove dihedral features from K-termini residues to avoid excessive noise"""
    if isinstance(feat_list, list):
        # List of feat_tuple
        assert len(feat_list[0]) == 2
        return [filter_termini_dihedrals(feat_list=feat, k=k) for feat in tqdm(feat_list)]
    else:
        # singe feat tuple
        assert len(feat_list) == 2
        feat, feat_name = feat_list
        res_idx = [int(name_.replace(')', '').split(' ')[-1]) for name_ in feat_name] # dihedral feat naming: e.g. "SIN(PHI 0 THR 333)""
        res_min = min(res_idx) + k # included
        res_max = max(res_idx) - k # included
        mask = [idx >= res_min and idx <= res_max for idx in res_idx]
        return feat[:, mask], list(np.array(feat_name)[mask])

