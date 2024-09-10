"""Utilities for calculating all atom representations.
-----------------------------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

import torch
from openfold.np import residue_constants as rc
from openfold.utils.tensor_utils import batched_gather

"""Construct denser atom positions (14 dimensions instead of 37)."""
restype_atom37_to_atom14 = []

for rt in rc.restypes:
    atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
    atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
    restype_atom37_to_atom14.append(
        [
            (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
            for name in rc.atom_types
        ]
    )

restype_atom37_to_atom14.append([0, 1] + [0] * 35)
restype_atom37_to_atom14 = torch.tensor(
    restype_atom37_to_atom14,
    dtype=torch.int32,
)

restype_atom37_mask = torch.zeros([21, 37], dtype=torch.float32)

for restype, restype_letter in enumerate(rc.restypes):
    restype_name = rc.restype_1to3[restype_letter]
    atom_names = rc.residue_atoms[restype_name]
    for atom_name in atom_names:
        atom_type = rc.atom_order[atom_name]
        restype_atom37_mask[restype, atom_type] = 1

restype_atom37_mask[20][1] = 1


def atom14_to_atom37(atom14, aatype):

    residx_atom37_mask = restype_atom37_mask.to(atom14.device)[aatype.long()]
    residx_atom37_to_atom14 = restype_atom37_to_atom14.to(atom14.device)[aatype.long()]
    # protein["atom37_atom_exists"] = residx_atom37_mask

    atom37 = batched_gather(
        atom14,
        residx_atom37_to_atom14.long(),
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )
    atom37 = atom37 * residx_atom37_mask[..., None]
    return atom37, residx_atom37_mask
