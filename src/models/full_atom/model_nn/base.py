"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""
import torch
from torch import nn
from openfold.np import residue_constants as rc
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)


class FoldNet(nn.Module):
    def __init__(
        self,
        embedder,
        structure_module,
    ):
        super().__init__()
        self.embedder = embedder
        self.structure_module = structure_module

    def forward(
        self,
        aatype,
        padding_mask,
        t,
        rigids_t,  # unscaled
        rigids_mask,
        res_idx=None,
        pretrained_node_repr=None,
        pretrained_edge_repr=None,
        **kwargs,
    ):
        if res_idx is None:
            res_idx = torch.arange(aatype.shape[1], device=aatype.device).expand_as(
                aatype
            )
        res_idx = res_idx * padding_mask

        node_feat, edge_feat = self.embedder(
            padding_mask=padding_mask,
            t=t,
            res_idx=res_idx,
            rigids_t=rigids_t,
            pretrained_node_repr=pretrained_node_repr,
            pretrained_edge_repr=pretrained_edge_repr,
        )

        model_out = self.structure_module(
            rigids_t=rigids_t,  # (B, L, 7) # unscaled
            node_feat=node_feat,  # (B, L, node_emb_size)
            edge_feat=edge_feat,  # (B, L, L, edge_emb_size)
            node_mask=rigids_mask,  # (B, L)
            padding_mask=padding_mask,  # (B, L)
        )

        all_frames_to_global = self.torsion_angles_to_frames(
            model_out["pred_rigids_0"],
            model_out["pred_torsions"],
            aatype,
        )

        model_out["pred_atom14"] = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            torch.fmod(aatype, 20),
        )

        model_out["pred_sidechain_frames"] = all_frames_to_global.to_tensor_4x4()

        return model_out

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    rc.restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    rc.restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    rc.restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    rc.restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
