"""Use openMM for energy/force evaluation

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Literal, Optional

import os
import numpy as np
import pandas as pd
from pathlib import Path
from time import perf_counter
from argparse import ArgumentParser

import openmm
from openmm import unit, app, Platform
from openmm.app import PDBFile
from openfold.np.relax.amber_minimize import _add_restraints
from openfold.np.relax.cleanup import fix_pdb as _fix_pdb
from src.utils.protein.faspr import faspr_pack
from src.utils.misc.process import mp_imap_unordered


# Check OpenMM Platform
print("OpenMM available platforms:")
for i in range(Platform.getNumPlatforms()):
    print(Platform.getPlatform(i).getName())

# =============================================================================
# Constants
# =============================================================================

CPU_COUNT = os.cpu_count()

KJ_PER_MOL = unit.kilojoules_per_mole
KCAL_PER_MOL = unit.kilocalories_per_mole
ANG = unit.angstroms

# Force
KCAL_PER_MOL_NM = unit.kilocalories_per_mole / (unit.nano * unit.meter)

STIFFNESS_UNIT = KCAL_PER_MOL / (ANG ** 2)


# =============================================================================
# Functions
# =============================================================================


def fix_pdb_struct(pdb_fpath, output_root):
    """Use openfold's PDBFixer pipeline to fix pdb files"""

    fixing_info = {}
    with open(pdb_fpath, 'r') as handle:
        fixed_pdb_str = _fix_pdb(handle, fixing_info)
    output_root = Path(output_root)
    output_fpath = output_root.joinpath(Path(pdb_fpath).stem + '_pdbfixed.pdb')
    info_output_fpath = output_root.joinpath(Path(pdb_fpath).stem + '_pdbfixed.info')
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_fpath, 'w') as handle:
        handle.write(fixed_pdb_str)
    with open(info_output_fpath, 'w') as handle:
        for key, val in fixing_info.items():
            handle.write(f"{key}: {val}\n")

    return output_fpath


def get_energy_profile(system, simulation, ret={}):

    # Get all forces
    state = simulation.context.getState(getEnergy=True, getForces=True)
    ret['potential_energy'] = state.getPotentialEnergy() / KCAL_PER_MOL

    # Get energy profile
    for i, f in enumerate(system.getForces()):
        state = simulation.context.getState(getEnergy=True, groups={i})
        ret[f.getName()] = state.getPotentialEnergy() / KCAL_PER_MOL

    if 'CustomExternalForce' in ret.keys():
        ret['PotentialEnergy'] = ret['potential_energy'] - ret['CustomExternalForce']
    return ret


def clip_force(force_np, max_force=1e2, esp=1e-8):
    """Clip forces by rescale large forces to length of max_force vector"""
    norm  = np.linalg.norm(force_np, axis=-1)
    clipped_norm = np.clip(norm, 0, max_force)
    coef = clipped_norm / (norm + esp)
    return coef[...,None] * force_np


def get_atom_forces(simulation, topology):

    atom_forces = simulation.context.getState(getForces=True).getForces() / KCAL_PER_MOL_NM
    atom_forces = np.array(atom_forces)

    ca_forces = []
    atom_idx = 0
    for j, res in enumerate(topology._chains[0].residues()):
        for atom in res.atoms():
            if atom.name =='CA':
                ca_forces.append(atom_forces[atom_idx])
            atom_idx += 1
    return atom_forces, ca_forces


def openmm_eval_energy(
    pdb_fpath,
    forcefield: Literal['amber14', 'charmm36'] = 'amber14',
    fix_pdb=False,
    add_H=True,
    minimize_struct=True,
    rset: Optional[Literal['non_hydrogen', 'c_alpha']] = 'non_hydrogen',
    stiffness: float = 10,
    output_root=None,
    # control
    tolerance=2.39,
    maxIterations=1,
    n_threads=24,
    use_gpu=False,
): 
    start_t = perf_counter()

    prefix = Path(pdb_fpath).stem
    if output_root is None:
        output_root = Path(pdb_fpath).parent
    
    forcefield = app.ForceField(
        'charmm36.xml' if forcefield == 'charmm36' else 'amber14/protein.ff14SB.xml', 
        'implicit/gbn2.xml'
    )
    if use_gpu:
        platform = openmm.Platform.getPlatformByName('CUDA')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        platform = openmm.Platform.getPlatformByName('CPU')

    # step 1: fix pdb and setup system
    if fix_pdb:
        pdb_fpath = fix_pdb_struct(pdb_fpath, output_root=output_root)
    pdbfile = PDBFile(str(pdb_fpath))
    modeller = openmm.app.Modeller(pdbfile.topology, pdbfile.positions)
    if fix_pdb is False and add_H:
        modeller.addHydrogens(forcefield, pH=7.0)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=app.HBonds,
        soluteDielectric=1.0, solventDielectric=78.5 # that's default as well
    )

    ret = {}
    if minimize_struct:
        # minimize structure and report
        if stiffness is not None and stiffness > 0.:
            stiffness = stiffness * STIFFNESS_UNIT
            _add_restraints(system, reference_pdb=modeller, stiffness=stiffness, rset=rset, exclude_residues=[])

        # Setup force group
        for i, f in enumerate(system.getForces()):
            f.setForceGroup(i)
            
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds) # dummy intergrator. Not used in minimizeEnergy
        if not use_gpu and n_threads is not None:
            platform_prop = {'Threads': str(int(n_threads))}
        elif use_gpu:
            platform_prop = {'UseCpuPme': 'False',}
        else:
            platform_prop = None
        simulation = app.Simulation(
            modeller.topology, system, integrator, 
            platform = platform, 
            platformProperties=platform_prop
        )
        simulation.context.setPositions(modeller.positions)
        # eval before opt
        energy_res = get_energy_profile(system=system, simulation=simulation)
        ret.update({'before_' + key: val for key, val in energy_res.items()})

        min_start_t = perf_counter()
        simulation.minimizeEnergy(
            tolerance=tolerance * KCAL_PER_MOL,
            maxIterations=maxIterations
        )
        min_end_t = perf_counter()

        # eval after opt
        energy_res = get_energy_profile(system=system, simulation=simulation)
        ret.update({'opt_' + key: val for key, val in energy_res.items()})

        # Save opted coordinates
        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(output_root/f'{prefix}_opt.pdb', 'w') as f:
            PDBFile.writeFile(simulation.topology, positions, f)
        prefix = prefix + '_opt'
        atom_forces, ca_forces = get_atom_forces(simulation, modeller.topology)
        ret['minimize_cost'] = min_end_t - min_start_t
        
    else:
        # Evaluate energy and forces only
        # Setup force group
        for i, f in enumerate(system.getForces()):
            f.setForceGroup(i)

        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds) # dummy intergrator. Not used in minimizeEnergy
        simulation = app.Simulation(
            modeller.topology, system, integrator, platform = platform, 
        )
        simulation.context.setPositions(modeller.positions)
        ret.update(get_energy_profile(system=system, simulation=simulation))
        atom_forces, ca_forces = get_atom_forces(simulation, modeller.topology)
        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(output_root/f'{prefix}_eval.pdb', 'w') as f:
            PDBFile.writeFile(simulation.topology, positions, f)
        prefix = prefix + '_eval'
    
    end_t = perf_counter()
    
    # -------------------- Save -------------------- 
    # Save forces    
    np.save(output_root/f'{prefix}_force.npy', atom_forces)
    np.save(output_root/f'{prefix}_ca_force.npy', ca_forces)
    np.save(output_root/f'{prefix}_ca_force_clip.npy', clip_force(ca_forces, max_force=1e4, esp=1e-8))

    # Save atom labels
    atom_idx, residue_names, residue_ids, atom_names = zip(*[
        [ix, atom.residue.name, atom.residue.id, atom.name]
        for ix, atom in enumerate(modeller.topology.atoms())
        if atom.residue.name not in ['CL', 'NA', 'HOH']
    ])
    atom_names = pd.DataFrame({
        'atom_idx': list(atom_idx),
        'residue_name': list(residue_names),
        'residue_id': list(residue_ids),
        'atom_name': list(atom_names),
    })
    atom_names.to_csv(output_root/f'{prefix}_atom_names.csv', index=False)

    ret['total_cost'] = end_t - start_t
    pd.DataFrame([ret]).to_csv(output_root/f'{prefix}_energy.csv', index=False)
    return ret


def openmm_routine(
    pdb_fpath, output_root=None, init_w_faspr=False, input_root=None,
    raise_error=False,
    **kwargs
):
    """Evaluate the energy and CA force"""
    pdb_fpath = Path(pdb_fpath)
    fname = pdb_fpath.stem
    output_root = Path(output_root)
    if input_root is None:
        output_root = output_root
    else:
        output_root = output_root.joinpath(pdb_fpath.parent.relative_to(input_root)) # keep the reletive path

    if init_w_faspr:
        energy_fpath = output_root/f"{fname}_faspr_packed{'_opt' if kwargs.get('minimize_struct', False) else ''}_energy.csv"
    else:
        energy_fpath = output_root/f"{fname}{'_opt' if kwargs.get('minimize_struct', False) else ''}_energy.csv"

    if energy_fpath.exists():
        # load forces
        energy_info = pd.read_csv(energy_fpath).squeeze().to_dict()
    else:
        if init_w_faspr:
            faspr_packed_fpath = output_root/f"{fname}_faspr_packed.pdb"
            pdb_fpath = faspr_pack(pdb_fpath, faspr_packed_fpath)
        
        if raise_error:
            energy_info = openmm_eval_energy(
                pdb_fpath=pdb_fpath,
                output_root=output_root,
                **kwargs
            )
        else:
            try:
                energy_info = openmm_eval_energy(
                    pdb_fpath=pdb_fpath,
                    output_root=output_root,
                    **kwargs
                )
            except Exception as e:
                print(f"OpenMM error: {pdb_fpath}: {e}")
                energy_info = {}
    energy_info['fname'] = fname
    return energy_info


def process_folder(
    input_root, output_root=None, n_proc=1, max_sample=None, chains=None, worker_id=None, world_size=None, train_ratio=0.9, drop_ratio=0.01, **kwargs
):
    # -------------------- Get all pdbs to process --------------------
    input_root = Path(input_root)
    if output_root is None:
        output_root = input_root
    output_root = Path(output_root)
    
    def verify_valid_sample_pdb(fpath):
        """Check if a sample pdb fpath is a valid one"""
        rel_path = str(Path(fpath).relative_to(input_root))
        is_not_openmm = 'openmm' not in rel_path
        is_orig_conf = '_conf' not in fpath.name and 'traj' not in fpath.name and not Path(fpath).is_relative_to(output_root)
        if chains is not None:
            is_in_chains = any(chain_name in rel_path for chain_name in chains)
        else:
            is_in_chains = True
        return is_orig_conf and is_not_openmm and is_in_chains

    pdb_fpath_list = []
    for subdir in input_root.rglob('*'):
        if not subdir.is_dir() or 'openmm' in subdir.name or (chains is not None and subdir.name not in chains):
            continue
        pdb_fpath_list += [fpath for fpath in subdir.rglob('*.pdb') if verify_valid_sample_pdb(fpath)]

    # -------------------- Process --------------------
    
    if max_sample is not None:
        pdb_fpath_list = pdb_fpath_list[:max_sample]
    pdb_fpath_list = sorted(pdb_fpath_list)
    if worker_id is None:
        print(f"Processing {len(pdb_fpath_list)} pdb files")
    else:
        total = len(pdb_fpath_list)
        pdb_fpath_list = pdb_fpath_list[worker_id::world_size]
        print(f"[Worker {worker_id}] {total} pdb files found. Processing {len(pdb_fpath_list)} on this worker")

    all_res = mp_imap_unordered(
        func=openmm_routine, iter=pdb_fpath_list, n_proc=n_proc,
        output_root=output_root, input_root=input_root, **kwargs
    )
    all_res = pd.DataFrame(all_res)
    
    all_res.to_csv(output_root/'all_energy_info.csv', index=False)
    process_energy_data(df=all_res, output_root=output_root, train_ratio=train_ratio, drop_ratio=drop_ratio)
    return all_res


def process_energy_data(df, output_root,  train_ratio=0.9, drop_ratio=0.01):
    split_data = df['fname'].str.extract(r'(?P<chain_name>.*)_samples(?P<sample_id>\d+)')
    df = pd.concat([df, split_data], axis=1)
    df['rank'] = df.groupby('chain_name')['opt_PotentialEnergy'].rank(method='first')
    df['count'] = df.groupby('chain_name')['opt_PotentialEnergy'].transform('count')
    df['drop_percent'] = np.ceil(df['count'] * drop_ratio).astype(int)
    def filter_extremes(group):
        min_rank, max_rank = group['rank'].min(), group['rank'].max()
        threshold = group['drop_percent'].iloc[0]
        return group[(group['rank'] >= min_rank + threshold) & (group['rank'] <= max_rank - threshold)]
    df = df.groupby('chain_name').apply(filter_extremes).reset_index(drop=True)
    df = df.drop(columns=['rank', 'count', 'drop_percent'])
    df['normalized_energy'] = df.groupby('chain_name')['opt_PotentialEnergy'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))


    train_list = []
    val_list = []

    for chain_name, group in df.groupby('chain_name'):
        n_train = int(np.floor(train_ratio * len(group)))  
        train = group.iloc[:n_train]
        val = group.iloc[n_train:]
        train_list.append(train)
        val_list.append(val)

    train_df = pd.concat(train_list)
    val_df = pd.concat(val_list)
    train_df.to_csv(output_root/'train.csv')
    val_df.to_csv(output_root/'val.csv')

# =============================================================================
# Classes
# =============================================================================

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input-root', type=str)
    parser.add_argument('--output-root', type=str)
    parser.add_argument('--init-w-faspr', type=lambda x: eval(x), default=False)
    parser.add_argument('--max-sample', type=int, default=None)
    parser.add_argument('--use-charmm36', type=lambda x: eval(x), default=False)
    parser.add_argument('--fix-pdb', type=lambda x: eval(x), default=True)
    parser.add_argument('--add-H', type=lambda x: eval(x), default=True)
    parser.add_argument('--minimize-struct', type=lambda x: eval(x), default=True)
    parser.add_argument('--fix-ca-only', type=lambda x: eval(x), default=False)
    parser.add_argument('--stiffness', type=float, default=10.)
    parser.add_argument('--tolerance', type=float, default=2.39)
    parser.add_argument('--max-iter', type=int, default=0)
    parser.add_argument('--n-proc', type=int, default=1)
    parser.add_argument('--n-threads', type=int, default=None)
    parser.add_argument('--use-gpu', type=lambda x: eval(x), default=False)
    parser.add_argument('--worker-id', type=int, default=None)
    parser.add_argument('--world-size', type=int, default=None)
    parser.add_argument('--chains', type=str, nargs='+', default=None)
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--drop-ratio', type=float, default=0.01)
    args = parser.parse_args()
    process_folder(
        input_root=args.input_root,
        output_root=args.output_root,
        init_w_faspr=args.init_w_faspr,
        max_sample=args.max_sample,
        n_proc=args.n_proc,
        # kwargs to openmm_eval_energy
        forcefield='charmm36' if args.use_charmm36 else 'amber14',
        fix_pdb=args.fix_pdb,
        add_H=args.add_H,
        minimize_struct=args.minimize_struct,
        rset='c_alpha' if args.fix_ca_only else 'non_hydrogen',
        stiffness=args.stiffness,
        tolerance=args.tolerance,
        maxIterations=args.max_iter,
        n_threads=args.n_threads,
        use_gpu=args.use_gpu,
        worker_id=args.worker_id,
        world_size=args.world_size,
        chains=args.chains,
        train_ratio=args.train_ratio,
        drop_ratio=args.drop_ratio,
    )
