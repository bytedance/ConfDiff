"""Analysis code modified from AlphaFlow (Bowen Jin, 2023).

https://github.com/bjing2016/alphaflow/blob/master/scripts/analyze_ensembles.py

----------------
MIT License

Copyright (c) 2024 Bowen Jing, Bonnie Berger, Tommi Jaakkola

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

----------------
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). 
All Bytedance's Modifications are Copyright (2024) Bytedance Ltd. and/or its affiliates. 
"""

# =============================================================================
# Imports
# =============================================================================

import argparse
from pathlib import Path
from time import perf_counter

from sklearn.decomposition import PCA
import mdtraj, pickle, warnings
import pandas as pd
import numpy as np

from scipy.optimize import linear_sum_assignment
import scipy.stats

from src.utils.misc.process import mp_imap

# =============================================================================
# Constants
# =============================================================================


REPORT_TAB_COLS = [
    'RMSF',
    'Pairwise RMSD',
    'Pairwise RMSD r',
    'Global RMSF r',
    'Per target RMSF r',
    'RMWD',
    'RMWD trans',
    'RMWD var',
    'MD PCA W2',
    'Joint PCA W2',
    'PC sim > 0.5 %',
    'Weak contacts J',
    'Transient contacts J',
    'Exposed residue J',
    'Exposed MI matrix rho'
]



# =============================================================================
#   Analysis functions
# =============================================================================


def get_pca(xyz):
    traj_reshaped = xyz.reshape(xyz.shape[0], -1)
    pca = PCA(n_components=min(traj_reshaped.shape))
    coords = pca.fit_transform(traj_reshaped)
    return pca, coords


def get_rmsds(traj1, traj2, broadcast=False):
    n_atoms = traj1.shape[1]
    traj1 = traj1.reshape(traj1.shape[0], n_atoms * 3)
    traj2 = traj2.reshape(traj2.shape[0], n_atoms * 3)
    if broadcast:
        traj1, traj2 = traj1[:,None], traj2[None]
    distmat = np.square(traj1 - traj2).sum(-1)**0.5 / n_atoms**0.5 * 10
    return distmat


def condense_sidechain_sasas(sasas, top):
    assert top.n_residues > 1

    if top.n_atoms != sasas.shape[1]:
        raise Exception(
            f"The number of atoms in top ({top.n_atoms}) didn't match the "
            f"number of SASAs provided ({sasas.shape[1]}). Make sure you "
            f"computed atom-level SASAs (mode='atom') and that you've passed "
            "the correct topology file and array of SASAs"
        )

    sc_mask = np.array([a.name not in ['CA', 'C', 'N', 'O', 'OXT'] for a in top.atoms])
    res_id = np.array([a.residue.index for a in top.atoms])
    
    rsd_sasas = np.zeros((sasas.shape[0], top.n_residues), dtype='float32')

    for i in range(top.n_residues):
        rsd_sasas[:, i] = sasas[:, sc_mask & (res_id == i)].sum(1)
    return rsd_sasas


def sasa_mi(sasa): 
    N, L = sasa.shape
    joint_probs = np.zeros((L, L, 2, 2))

    joint_probs[:,:,1,1] = (sasa[:,:,None] & sasa[:,None,:]).mean(0)
    joint_probs[:,:,1,0] = (sasa[:,:,None] & ~sasa[:,None,:]).mean(0)
    joint_probs[:,:,0,1] = (~sasa[:,:,None] & sasa[:,None,:]).mean(0)
    joint_probs[:,:,0,0] = (~sasa[:,:,None] & ~sasa[:,None,:]).mean(0)

    marginal_probs = np.stack([1-sasa.mean(0), sasa.mean(0)], -1)
    indep_probs = marginal_probs[None,:,None,:] * marginal_probs[:,None,:,None] 
    mi = np.nansum(joint_probs * np.log(joint_probs / indep_probs), (-1, -2))
    mi[np.arange(L), np.arange(L)] = 0
    return mi

       
def get_mean_covar(xyz):
    mean = xyz.mean(0)
    xyz = xyz - mean
    covar = (xyz[...,None] * xyz[...,None,:]).mean(0)
    return mean, covar


def sqrtm(M):
    D, P = np.linalg.eig(M)
    out = (P * np.sqrt(D[:,None])) @ np.linalg.inv(P)
    return out


def get_wasserstein(distmat, p=2):
    assert distmat.shape[0] == distmat.shape[1]
    distmat = distmat ** p
    row_ind, col_ind = linear_sum_assignment(distmat)
    return distmat[row_ind, col_ind].mean() ** (1/p)

def align_tops(top1, top2):
    names1 = [repr(a) for a in top1.atoms]
    names2 = [repr(a) for a in top2.atoms]

    intersection = [nam for nam in names1 if nam in names2]
    
    mask1 = [names1.index(nam) for nam in intersection]
    mask2 = [names2.index(nam) for nam in intersection]
    return mask1, mask2



# ************************************************************************
#   Report functions
# ************************************************************************


def correlations(a, b, prefix=''):
    return {
        prefix + 'pearson': scipy.stats.pearsonr(a, b)[0],
        prefix + 'spearman': scipy.stats.spearmanr(a, b)[0],
        prefix + 'kendall': scipy.stats.kendalltau(a, b)[0],
    }


def analyze_data(data):
    mi_mats = {}
    df = []
    for name, out in data.items():
        item = {
            'name': name,
            'md_pairwise': out['ref_mean_pairwise_rmsd'],
            'af_pairwise': out['af_mean_pairwise_rmsd'],
            'cosine_sim': abs(out['cosine_sim']),
            'emd_mean': np.square(out['emd_mean']).mean() ** 0.5,
            'emd_var': np.square(out['emd_var']).mean() ** 0.5,
        } | correlations(out['af_rmsf'], out['ref_rmsf'], prefix='rmsf_')
        if 'EMD,ref' not in out:
            out['EMD,ref'] = out['EMD-2,ref']
            out['EMD,af2'] = out['EMD-2,af2']
            out['EMD,joint'] = out['EMD-2,joint']
        for emd_dict, emd_key in [
            (out['EMD,ref'], 'ref'),
            (out['EMD,joint'], 'joint')
        ]:
            item.update({
                emd_key + 'emd': emd_dict['ref|af'],
                emd_key + 'emd_tr': emd_dict['ref mean|af mean'],
                emd_key + 'emd_int': (emd_dict['ref|af']**2 - emd_dict['ref mean|af mean']**2)**0.5,
            })
    
        try:
            crystal_contact_mask = out['crystal_distmat'] < 0.8
            ref_transient_mask = (~crystal_contact_mask) & (out['ref_contact_prob'] > 0.1)
            af_transient_mask = (~crystal_contact_mask) & (out['af_contact_prob'] > 0.1)
            ref_weak_mask = crystal_contact_mask & (out['ref_contact_prob'] < 0.9)
            af_weak_mask = crystal_contact_mask & (out['af_contact_prob'] < 0.9)
            item.update({
                'weak_contacts_iou': (ref_weak_mask & af_weak_mask).sum() / (ref_weak_mask | af_weak_mask).sum(),
                'transient_contacts_iou': (ref_transient_mask & af_transient_mask).sum() / (ref_transient_mask | af_transient_mask).sum() 
            })
        except:
            item.update({
                'weak_contacts_iou': np.nan,
                'transient_contacts_iou': np.nan, 
            })
        sasa_thresh = 0.02
        buried_mask = out['crystal_sasa'][0] < sasa_thresh
        ref_sa_mask = (out['ref_sa_prob'] > 0.1) & buried_mask
        af_sa_mask = (out['af_sa_prob'] > 0.1) & buried_mask
    
        item.update({
            'num_sasa': ref_sa_mask.sum(),
            'sasa_iou': (ref_sa_mask & af_sa_mask).sum() / (ref_sa_mask | af_sa_mask).sum(),
        })
        item.update(correlations(out['ref_mi_mat'].flatten(), out['af_mi_mat'].flatten(), prefix='exposon_mi_'))
       
        df.append(item)
    df = pd.DataFrame(df).set_index('name')#.join(val_df)
    all_ref_rmsf = np.concatenate([data[name]['ref_rmsf'] for name in df.index])
    all_af_rmsf = np.concatenate([data[name]['af_rmsf'] for name in df.index])
    return all_ref_rmsf, all_af_rmsf, df, data


def report_analysis(result_root):
    
    datas = {}
    for exp_name, data_pkl in result_root.items():
        if not Path(data_pkl).is_file():
            data_pkl = f"{data_pkl}/aflow_analysis.pkl"
        with open(data_pkl, 'rb') as f:
            data = pickle.load(f)
        data = {chain_name: result for chain_name, result in data}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            datas[exp_name] = analyze_data(data)

    new_df = []
    for key in datas:
        ref_rmsf, af_rmsf, df, data = datas[key]
        print(f'[{key}] {len(df)} cases')
        new_df.append({
            'exp': key,
            'count': len(df),
            'MD pairwise RMSD': df.md_pairwise.median(),
            'Pairwise RMSD': df.af_pairwise.median(),
            'Pairwise RMSD r': scipy.stats.pearsonr(df.md_pairwise, df.af_pairwise)[0],
            'MD RMSF': np.median(ref_rmsf),
            'RMSF': np.median(af_rmsf),
            'Global RMSF r': scipy.stats.pearsonr(ref_rmsf, af_rmsf)[0],
            'Per target RMSF r': df.rmsf_pearson.median(),
            'RMWD': np.sqrt(df.emd_mean**2 + df.emd_var**2).median(),
            'RMWD trans': df.emd_mean.median(),
            'RMWD var': df.emd_var.median(),
            'MD PCA W2': df.refemd.median(),
            'Joint PCA W2': df.jointemd.median(),
            'PC sim > 0.5 %': (df.cosine_sim > 0.5).mean() * 100,
            'Weak contacts J': df.weak_contacts_iou.median(),
            'Weak contacts nans': df.weak_contacts_iou.isna().mean(),
            'Transient contacts J': df.transient_contacts_iou.median(),
            'Transient contacts nans': df.transient_contacts_iou.isna().mean(),
            'Exposed residue J': df.sasa_iou.median(),
            'Exposed MI matrix rho': df.exposon_mi_spearman.median(),
        })

    new_df = pd.DataFrame(new_df).set_index('exp')
    return {
        'report_tab': new_df,
        'all_results': datas
    }


# =============================================================================
# Main
# =============================================================================

 
def eval_one_protein(
    chain_name,
    result_dir,
    atlas_dir,
    bb_only=False,
    ca_only=False,
):
    """Evaluate one protein ensemble results"""
    print('Analyzing', chain_name)
    start_t = perf_counter()

    # -------------------- Load data and preprocess --------------------
    # load reference
    atlas_dir = Path(atlas_dir)
    topfile = str(atlas_dir/chain_name/f'{chain_name}.pdb')
    ref_aa = mdtraj.load(topfile)
    traj_aa = mdtraj.load(f'{atlas_dir}/{chain_name}/{chain_name}_prod_R1_fit.xtc', top=topfile) \
        + mdtraj.load(f'{atlas_dir}/{chain_name}/{chain_name}_prod_R2_fit.xtc', top=topfile) \
        + mdtraj.load(f'{atlas_dir}/{chain_name}/{chain_name}_prod_R3_fit.xtc', top=topfile)
    # load results
    if Path(result_dir).joinpath(chain_name).exists():
        # is sub folder based
        aftraj_aa = mdtraj.load([
            str(pdb_fpath) for pdb_fpath in Path(result_dir).joinpath(chain_name).glob(f'{chain_name}*.pdb')
        ])
    else:
        assert Path(result_dir).joinpath(f"{chain_name}.pdb").exists()
        aftraj_aa = mdtraj.load(f"{result_dir}/{chain_name}.pdb")

    # Remove H
    traj_aa.atom_slice([a.index for a in traj_aa.top.atoms if a.element.symbol != 'H'], True)
    ref_aa.atom_slice([a.index for a in ref_aa.top.atoms if a.element.symbol != 'H'], True)
    aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.element.symbol != 'H'], True)
    print(
        f'[{chain_name}] data loaded, hydrogen removed: \n'
        f'  - Reference: {traj_aa.n_frames} frames, {traj_aa.n_atoms} atoms\n'
        f'  - Crystal:   {ref_aa.n_frames} frames, {ref_aa.n_atoms} atoms\n'
        f'  - Sample:    {aftraj_aa.n_frames} frames, {aftraj_aa.n_atoms} atoms'
    )

    # -------------------- Preprocess --------------------
    if bb_only:
        aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.name in ['CA', 'C', 'N', 'O', 'OXT']], True)
        print(f'[{chain_name}] backbone Only, removing sidechains, sample conformations have {aftraj_aa.n_atoms} atoms')
    elif ca_only:
        aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.name == 'CA'], True)
        print(f'[{chain_name}] CA Only, removing other atoms, sample conformations have {aftraj_aa.n_atoms} atoms')    
    
    refmask, afmask = align_tops(traj_aa.top, aftraj_aa.top) # align topology to get shared atoms
    traj_aa.atom_slice(refmask, True)
    ref_aa.atom_slice(refmask, True)
    aftraj_aa.atom_slice(afmask, True)
    print(f'[{chain_name}] aligned on {aftraj_aa.n_atoms} atoms')

    # -------------------- CA PCA --------------------
    out = {}
    np.random.seed(137 + sum([ord(c) for c in chain_name])) # varying seed for different chains
    RAND1 = np.random.randint(0, traj_aa.n_frames, aftraj_aa.n_frames) # downsample to sample ensemble
    RAND2 = np.random.randint(0, traj_aa.n_frames, aftraj_aa.n_frames)
    RAND1K = np.random.randint(0, traj_aa.n_frames, 1000)

    traj_aa.superpose(ref_aa)
    aftraj_aa.superpose(ref_aa)

    out['ref_atom_names'] = [repr(a) for a in traj_aa.top.atoms]
    out['af_atom_names'] = [repr(a) for a in aftraj_aa.top.atoms]
    out['ca_mask'] = ca_mask = [a.index for a in traj_aa.top.atoms if a.name == 'CA']
    traj = traj_aa.atom_slice(ca_mask, False)
    ref = ref_aa.atom_slice(ca_mask, False)
    # FIX: traj_aa.top.atoms != aftraj_aa.top.atoms. atom_slice does not change atom orders
    af_ca_mask = [a.index for a in aftraj_aa.top.atoms if a.name == 'CA']
    aftraj = aftraj_aa.atom_slice(af_ca_mask, False)
    print(f'Sliced {aftraj.n_atoms} C-alphas')

    traj.superpose(ref)
    aftraj.superpose(ref)
    
    n_atoms = aftraj.n_atoms

    print(f'Doing PCA')

    ref_pca, ref_coords = get_pca(traj.xyz)
    af_coords_ref_pca = ref_pca.transform(aftraj.xyz.reshape(aftraj.n_frames, -1))
    seed_coords_ref_pca = ref_pca.transform(ref.xyz.reshape(1, -1))
    out['ref_pca'] = ref_pca
    out['ref_pca_proj_ref2d'] = ref_coords[:, :2]
    out['ref_pca_proj_af2d'] = af_coords_ref_pca[:, :2]
    out['ref_pca_proj_seed2d'] = seed_coords_ref_pca[:, :2]
    
    af_pca, af_coords = get_pca(aftraj.xyz)
    ref_coords_af_pca = af_pca.transform(traj.xyz.reshape(traj.n_frames, -1))
    seed_coords_af_pca = af_pca.transform(ref.xyz.reshape(1, -1))
    out['af_pca'] = af_pca
    out['af_pca_proj_ref2d'] = ref_coords_af_pca[:, :2]
    out['af_pca_proj_af2d'] = af_coords[:, :2]
    out['af_pca_proj_seed2d'] = seed_coords_af_pca[:, :2]
    
    joint_pca, _ = get_pca(np.concatenate([traj[RAND1].xyz, aftraj.xyz]))
    af_coords_joint_pca = joint_pca.transform(aftraj.xyz.reshape(aftraj.n_frames, -1))
    ref_coords_joint_pca = joint_pca.transform(traj.xyz.reshape(traj.n_frames, -1))
    seed_coords_joint_pca = joint_pca.transform(ref.xyz.reshape(1, -1))
    out['joint_pca'] = joint_pca
    out['joint_pca_proj_ref2d'] = ref_coords_joint_pca[:, :2]
    out['joint_pca_proj_af2d'] = af_coords_joint_pca[:, :2]
    out['joint_pca_proj_seed2d'] = seed_coords_joint_pca[:, :2]
    
    out['ref_variance'] = ref_pca.explained_variance_ / n_atoms * 100
    out['af_variance'] = af_pca.explained_variance_ / n_atoms * 100
    out['joint_variance'] = joint_pca.explained_variance_ / n_atoms * 100
    out['ref_variance_ratio'] = ref_pca.explained_variance_ratio_
    out['af_variance_ratio'] = af_pca.explained_variance_ratio_
    out['joint_variance_ratio'] = joint_pca.explained_variance_ratio_

    # -------------------- Coords RMSF --------------------
    out['af_rmsf'] = mdtraj.rmsf(aftraj_aa, ref_aa) * 10
    out['ref_rmsf'] = mdtraj.rmsf(traj_aa, ref_aa) * 10
    
    print(f'Computing atomic EMD')
    ref_mean, ref_covar = get_mean_covar(traj_aa[RAND1K].xyz)
    af_mean, af_covar = get_mean_covar(aftraj_aa.xyz)
    out['emd_mean'] = (np.square(ref_mean - af_mean).sum(-1) ** 0.5) * 10
    try:
        out['emd_var'] = (np.trace(ref_covar + af_covar - 2*sqrtm(ref_covar @ af_covar), axis1=1,axis2=2) ** 0.5) * 10
    except:
        out['emd_var'] = np.trace(ref_covar) ** 0.5 * 10

    # -------------------- SASA --------------------
    print(f'Analyzing SASA')
    sasa_thresh = 0.02
    af_sasa = mdtraj.shrake_rupley(aftraj_aa, probe_radius=0.28)
    af_sasa = condense_sidechain_sasas(af_sasa, aftraj_aa.top)
    ref_sasa = mdtraj.shrake_rupley(traj_aa[RAND1K], probe_radius=0.28)
    ref_sasa = condense_sidechain_sasas(ref_sasa, traj_aa.top)
    crystal_sasa = mdtraj.shrake_rupley(ref_aa, probe_radius=0.28)
    out['crystal_sasa'] = condense_sidechain_sasas(crystal_sasa, ref_aa.top)
    
    out['ref_sa_prob'] = (ref_sasa > sasa_thresh).mean(0)
    out['af_sa_prob'] = (af_sasa > sasa_thresh).mean(0)
    out['ref_mi_mat'] = sasa_mi(ref_sasa > sasa_thresh)
    out['af_mi_mat'] = sasa_mi(af_sasa > sasa_thresh)
    
    # -------------------- Contacts --------------------
    ref_distmat = np.linalg.norm(traj[RAND1].xyz[:,None,:] - traj[RAND1].xyz[:,:,None], axis=-1)
    af_distmat = np.linalg.norm(aftraj.xyz[:,None,:] - aftraj.xyz[:,:,None], axis=-1)

    out['ref_contact_prob'] = (ref_distmat < 0.8).mean(0)
    out['af_contact_prob'] = (af_distmat < 0.8).mean(0)
    out['crystal_distmat'] = np.linalg.norm(ref.xyz[0,None,:] - ref.xyz[0,:,None], axis=-1)
    
    out['ref_mean_pairwise_rmsd'] = get_rmsds(traj[RAND1].xyz, traj[RAND2].xyz, broadcast=True).mean()
    out['af_mean_pairwise_rmsd'] = get_rmsds(aftraj.xyz, aftraj.xyz, broadcast=True).mean()

    out['ref_rms_pairwise_rmsd'] = np.square(get_rmsds(traj[RAND1].xyz, traj[RAND2].xyz, broadcast=True)).mean() ** 0.5
    out['af_rms_pairwise_rmsd'] = np.square(get_rmsds(aftraj.xyz, aftraj.xyz, broadcast=True)).mean() ** 0.5

    out['ref_self_mean_pairwise_rmsd'] = get_rmsds(traj[RAND1].xyz, traj[RAND1].xyz, broadcast=True).mean()
    out['ref_self_rms_pairwise_rmsd'] = np.square(get_rmsds(traj[RAND1].xyz, traj[RAND1].xyz, broadcast=True)).mean() ** 0.5
    
    out['cosine_sim'] = (ref_pca.components_[0] * af_pca.components_[0]).sum() 
    

    def get_emd(ref_coords1, ref_coords2, af_coords, seed_coords, K=None):
        if len(ref_coords1.shape) == 3:
            ref_coords1 = ref_coords1.reshape(ref_coords1.shape[0], -1)
            ref_coords2 = ref_coords2.reshape(ref_coords2.shape[0], -1)
            af_coords = af_coords.reshape(af_coords.shape[0], -1)
            seed_coords = seed_coords.reshape(seed_coords.shape[0], -1)
        if K is not None:
            ref_coords1 = ref_coords1[:,:K]
            ref_coords2 = ref_coords2[:,:K]
            af_coords = af_coords[:,:K]
            seed_coords = seed_coords[:,:K]
        emd = {}
        emd['ref|ref mean'] = (np.square(ref_coords1 - ref_coords1.mean(0)).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10
        
        # W2 between MD and MD
        distmat = np.square(ref_coords1[:,None] - ref_coords2[None]).sum(-1) 
        distmat = distmat ** 0.5 / n_atoms ** 0.5 * 10
        emd['ref|ref2'] = get_wasserstein(distmat)
        emd['ref mean|ref2 mean'] = np.square(ref_coords1.mean(0) - ref_coords2.mean(0)).sum() ** 0.5 / n_atoms ** 0.5 * 10
        
        # W2 between AF and MD
        distmat = np.square(ref_coords1[:,None] - af_coords[None]).sum(-1) 
        distmat = distmat ** 0.5 / n_atoms ** 0.5 * 10
        emd['ref|af'] = get_wasserstein(distmat)

        # Distance of dist center
        emd['ref mean|af mean'] = np.square(ref_coords1.mean(0) - af_coords.mean(0)).sum() ** 0.5 / n_atoms ** 0.5 * 10

        # mean dist to seed
        emd['ref|seed'] = (np.square(ref_coords1 - seed_coords).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10
        emd['ref mean|seed'] = (np.square(ref_coords1.mean(0) - seed_coords).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10

        emd['af|seed'] = (np.square(af_coords - seed_coords).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10
        emd['af mean|seed'] = (np.square(af_coords.mean(0) - seed_coords).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10

        # Variance
        emd['af|af mean'] = (np.square(af_coords - af_coords.mean(0)).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10

        return emd

    # Earth Mover Distance
    print("Computing PCA EMD ...")
    K=2
    out[f'EMD,ref'] = get_emd(ref_coords[RAND1], ref_coords[RAND2], af_coords_ref_pca, seed_coords_ref_pca, K=K)
    out[f'EMD,af2'] = get_emd(ref_coords_af_pca[RAND1], ref_coords_af_pca[RAND2], af_coords, seed_coords_af_pca, K=K)
    out[f'EMD,joint'] = get_emd(ref_coords_joint_pca[RAND1], ref_coords_joint_pca[RAND2], af_coords_joint_pca, seed_coords_joint_pca, K=K)

    end_t = perf_counter()
    print(f'[{chain_name}] eval finished in {end_t - start_t:.1f} seconds')

    return chain_name, out


def eval_all_proteins(
    result_dir, atlas_dir, chain_name_list = None,
    n_proc=1, bb_only=False, ca_only=False
):

    if chain_name_list in [None, []]:
        chain_name_list = [subdir.stem for subdir in Path(result_dir).glob('*') if subdir.is_dir()]
    if len(chain_name_list) == 0:
        # single PDB file for ensembles
        chain_name_list = [pdb_file.stem for pdb_file in Path(result_dir).glob('*.pdb') if '.xtc' not in pdb_file.name]
    if len(chain_name_list) == 0:
        raise FileNotFoundError(f"Result not found under {result_dir}")
    out = mp_imap(
        func=eval_one_protein, iter=chain_name_list, n_proc=n_proc,
        result_dir=result_dir, atlas_dir=atlas_dir, bb_only=bb_only, ca_only=ca_only
    )

    with open(f"{result_dir}/aflow_analysis.pkl", 'wb') as f:
        f.write(pickle.dumps(out))


# ************************************************************************
#   Report functions
# ************************************************************************

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, required=True)
    parser.add_argument('--atlas-dir', type=str, required=True)
    parser.add_argument('--pdb-id', nargs='*', default=[])
    parser.add_argument('--bb-only', action='store_true')
    parser.add_argument('--ca-only', action='store_true')
    parser.add_argument('--n-proc', type=int, default=1)

    args = parser.parse_args()

    eval_all_proteins(
        args.result_dir, atlas_dir=args.atlas_dir, chain_name_list=args.pdb_id,
        n_proc=args.n_proc, bb_only=args.bb_only, ca_only=args.ca_only
    )
