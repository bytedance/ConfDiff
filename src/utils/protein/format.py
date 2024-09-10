"""Convert protein file formats

----------------
Copyright (2023) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
from collections import deque
import numpy as np
import pandas as pd


from Bio.PDB import Structure
from Bio.PDB import PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder

# =============================================================================
# Constants
# =============================================================================


ELEMENT_ID_TO_ATOM_TYPE = {
    '1': 'H',
    '6': 'C',
    '7': 'N',
    '8': 'O',
    '16': 'S',
    '15': 'P',
}


ATOM_TEMPLATE = "ATOM  {atom_id:>5} {atom_name:4s} {res_type:>3s} {chain_name:1s}{res_id:>4}    {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}      {seg_id:>4s}{element:>2}  "
TER_TEMPLATE = "TER   {atom_id:>5} {atom_name:4s} {res_type:>3s} {chain_name:1s}{res_id:>4}                                                      "


# =============================================================================
# Functions
# =============================================================================


def mae_to_pdb(mae_fpath, pdb_fpath) -> Structure:
    """Convert a MAE topology file to a pdb topology file
    
    Args:
        mae_fpath: Path to the MAE file
        pdb_fpath: Path to the PDB file
    """

    with open(mae_fpath, 'r') as handle:
        line_que = deque([l.strip('\n') for l in handle.readlines()])
    
    line = line_que.popleft()
    while "m_atom" not in line:
        line = line_que.popleft()
    
    # seek to n_atom
    num_atoms = int(line[line.find('[') + 1: line.find(']')])
    
    # parse column names
    col_names = ['line']
    line = line_que.popleft()
    while ":::" not in line:
        if line.strip() != '':
            col_names.append(line.strip('\n').strip())
        line = line_que.popleft()
    
    # parse atom info
    atom_infos = []
    line = line_que.popleft()
    while ":::" not in line:
        parts = ['']
        in_quote = False
        for ch in line:
            if ch == '"':
                in_quote = not in_quote
            elif ch == ' ':
                if in_quote:
                    parts[-1] += ' '
                elif parts[-1] != '':
                    parts.append('')
            else:
                parts[-1] += ch
        
        if parts[-1] == '':
            parts = parts[:-1]
        assert len(parts) == len(col_names), f"{len(parts)} vs {len(col_names)}: error in line {line}"
        atom_infos.append(parts)
        line = line_que.popleft()
    assert len(atom_infos) == num_atoms, f"number of atoms mismatch: {num_atoms} atoms vs {len(atom_infos)} rows"
    
    atom_info = pd.DataFrame(atom_infos, columns=col_names)

    # write pdb
    with open(pdb_fpath, 'w') as handle:

        for ix, row in atom_info.iterrows():
            formatted = ATOM_TEMPLATE.format(
                atom_id=row['line'],
                atom_name=row['s_m_pdb_atom_name'],
                res_type=row['s_m_pdb_residue_name'].strip(),
                chain_name=row['s_m_chain_name'],
                res_id=row['i_m_residue_number'],
                x=float(row['r_m_x_coord'].strip()),
                y=float(row['r_m_y_coord'].strip()),
                z=float(row['r_m_z_coord'].strip()),
                occupancy=1.0,
                b_factor=1.0,
                seg_id=row['s_m_pdb_segment_name'].strip(),
                element=ELEMENT_ID_TO_ATOM_TYPE[row['i_m_atomic_number']]
            )
            assert len(formatted) == 80, f"Wrong length: '{formatted}' ({len(formatted)})"

            handle.write(formatted + '\n')
        
        # TER
        ter_line = TER_TEMPLATE.format(
            atom_id=int(row['line']) + 1,
            res_type=row['s_m_pdb_residue_name'].strip(),
            chain_name=row['s_m_chain_name'],
            res_id=row['i_m_residue_number'],
            atom_name=''
        )
        assert len(ter_line) == 80, f"Wrong length: '{ter_line}' ({len(ter_line)})"
        handle.write(ter_line + '\n')
        handle.write('END\n')

    return atom_info




# =============================================================================
# Classes
# =============================================================================

