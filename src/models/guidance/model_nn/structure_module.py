"""Fork of Openfold's IPA."""

import torch
import torch.nn as nn

from src.models.full_atom.model_nn.structure_module import (
    StructureModule,
    StructureModuleTransition,
    BackboneUpdate,
    InvariantPointAttention,
    Linear,
)


class MultiIPALayer(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_hidden,
        c_skip,
        no_heads,
        no_qk_points,
        no_v_points,
        num_layers,
    ):

        super(MultiIPALayer, self).__init__()
        self.ipa_dropout = nn.Dropout(0.1)
        self.layer_norm_ipa = nn.LayerNorm(c_s)

        self.transition = StructureModuleTransition(
            c_s,
        )
        self.bb_update = BackboneUpdate(c_s)

        self.layers = nn.ModuleDict()
        for b in range(num_layers):
            self.layers[f"ipa_{b}"] = InvariantPointAttention(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                c_skip=c_skip,
                no_heads=no_heads,
                no_qk_points=no_qk_points,
                no_v_points=no_v_points,
            )

        self.num_layers = num_layers

    def forward(self, r, s, z, mask):
        for b in range(self.num_layers):
            # IPA
            ipa_embed = self.layers[f"ipa_{b}"](r=r, s=s, z=z, mask=mask)
            ipa_embed = ipa_embed * mask[..., None]
            s = s + ipa_embed
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            r = r.compose_q_update_vec(self.bb_update(s))

        return r
        # return {

        # }


class EnergyIPA(StructureModule):
    def __init__(
        self,
        num_ipa_blocks: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        c_skip: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        seq_tfmr_num_heads: int,
        seq_tfmr_num_layers: int,
    ):
        super(EnergyIPA, self).__init__(
            num_ipa_blocks,
            c_s,
            c_z,
            c_hidden,
            c_skip,
            no_heads,
            no_qk_points,
            no_v_points,
            seq_tfmr_num_heads,
            seq_tfmr_num_layers,
        )

        self.pred_energy_t_net = StructureModuleTransition(c_s)
        self.energy_t_final = Linear(c_s, 1, init="final")
        # self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        rigids_t: torch.Tensor,  # (B, L, 7)
        node_feat: torch.Tensor,  # (B, L, node_emb_size)
        edge_feat: torch.Tensor,  # (B, L, L, edge_emb_size)
        node_mask: torch.Tensor,  # (B, L)
        padding_mask: torch.Tensor,  # (B, L)
    ):
        model_out = super(EnergyIPA, self).forward(
            rigids_t=rigids_t,
            node_feat=node_feat,
            edge_feat=edge_feat,
            node_mask=node_mask,
            padding_mask=padding_mask,
            return_feat=True,
        )

        pred_energy_t = self.pred_energy_t_net(model_out["node_feat"])
        pred_energy_t = self.energy_t_final(pred_energy_t.mean(dim=1))

        model_out["pred_energy_t"] = pred_energy_t

        return model_out


class ForceIPA(StructureModule):
    def __init__(
        self,
        num_ipa_blocks: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        c_skip: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        seq_tfmr_num_heads: int,
        seq_tfmr_num_layers: int,
        pred_force_0: bool = False,
    ):
        super(ForceIPA, self).__init__(
            num_ipa_blocks,
            c_s,
            c_z,
            c_hidden,
            c_skip,
            no_heads,
            no_qk_points,
            no_v_points,
            seq_tfmr_num_heads,
            seq_tfmr_num_layers,
        )

        # torsion angle prediction
        # self.torsion_pred = TorsionAngles(c_s, 1)

        self.pred_force_t_net = MultiIPALayer(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            c_skip=c_skip,
            no_heads=no_heads,
            no_qk_points=no_qk_points,
            no_v_points=no_v_points,
            num_layers=6,
        )
        # self.use_beta = use_beta
        self.pred_force_0 = pred_force_0

        if pred_force_0:
            self.pred_force_0_net = MultiIPALayer(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                c_skip=c_skip,
                no_heads=no_heads,
                no_qk_points=no_qk_points,
                no_v_points=no_v_points,
                num_layers=6,
            )

    def forward(
        self,
        rigids_t: torch.Tensor,  # (B, L, 7)
        node_feat: torch.Tensor,  # (B, L, node_emb_size)
        edge_feat: torch.Tensor,  # (B, L, L, edge_emb_size)
        node_mask: torch.Tensor,  # (B, L)
        padding_mask: torch.Tensor,  # (B, L)
    ):
        model_out = super(ForceIPA, self).forward(
            rigids_t=rigids_t,
            node_feat=node_feat,
            edge_feat=edge_feat,
            node_mask=node_mask,
            padding_mask=padding_mask,
            return_feat=True,
        )

        model_out["pred_force_t"] = self.pred_force_t_net(
            r=model_out["curr_rigids"],
            s=model_out["node_feat"],
            z=model_out["edge_feat"],
            mask=node_mask,
        ).get_trans()
        if self.pred_force_0:
            model_out["pred_force_0"] = self.pred_force_0_net(
                r=model_out["curr_rigids"],
                s=model_out["node_feat"],
                z=model_out["edge_feat"],
                mask=node_mask,
            ).get_trans()
        return model_out
