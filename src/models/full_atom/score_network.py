from typing import Dict, Tuple, Any
import torch
from torch import nn
from openfold.utils import rigid_utils as ru
from .utils.all_atom import atom14_to_atom37
from openfold.config import model_config
import openfold.utils.loss as af2_loss
from tqdm import tqdm


class BaseScoreNetwork(nn.Module):
    def __init__(
        self,
        model_nn,
        cfg,
        **kwargs,
    ):
        super().__init__()

        self.model_nn = model_nn
        self.diffuser = None
        self.cfg = cfg

        # AlphaFold config
        af2_config = model_config("model_3_ptm", train=True, low_prec=False)
        af2_config.data.common.use_templates = False
        af2_config.data.max_recycling_iters = 0
        self.af2_config = af2_config
    
    @property
    def device(self):
        return self.model_nn.device

    def forward(
        self,
        aatype,
        t,
        rigids_t,
        rigids_mask,
        padding_mask,
        gt_feat,
        res_idx=None,
        pretrained_node_repr=None,
        pretrained_edge_repr=None,
        **kwargs,
    ):
        # self.diffuser._so3_diffuser.use_cached_score = False # see cfg, use_cached_score is turned off
        rigids_t = ru.Rigid.from_tensor_7(rigids_t)
        rigids_mask = rigids_mask * padding_mask

        """ Model predicts rigids_0 and torsion angles. """
        model_out = self.model_nn(
            aatype=aatype,
            padding_mask=padding_mask,
            t=t,
            rigids_t=rigids_t,
            rigids_mask=rigids_mask,
            res_idx=res_idx,
            pretrained_node_repr=pretrained_node_repr,
            pretrained_edge_repr=pretrained_edge_repr,
        )

        pred_rot_score = (
            self.diffuser.calc_rot_score(
                rigids_t.get_rots(),
                model_out["pred_rigids_0"].get_rots(),
                t,
            )
            * rigids_mask[..., None]
        )

        pred_trans_score = (
            self.diffuser.calc_trans_score(
                rigids_t.get_trans(),
                model_out["pred_rigids_0"].get_trans(),
                t[:, None, None],
                use_torch=True,
            )
            * rigids_mask[..., None]
        )

        loss, aux_info = self.loss_fn(
            t,
            aatype,
            rigids_mask,
            model_out["pred_rigids_0"],
            model_out["pred_torsions"],
            model_out["pred_atom14"],
            # pred_atom37,
            pred_rot_score,
            pred_trans_score,
            model_out["pred_sidechain_frames"],
            gt_feat,
        )
        return loss, aux_info

    def loss_fn(
        self,
        t,
        aatype,
        rigids_mask,
        pred_rigids_0,
        pred_torsions,
        pred_atom14,
        pred_rot_score,
        pred_trans_score,
        pred_sidechain_frame,
        gt_feat,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # rotation score
        gt_rot_score = gt_feat["rot_score"]  # (B, L, 3)
        gt_rot_score_norm = gt_feat["rot_score_norm"]  # (B,)
        # translation score
        gt_trans_score = gt_feat["trans_score"]  # (B, L, 3)
        gt_trans_score_norm = gt_feat["trans_score_norm"]  # (B,)
        gt_bb_coords = gt_feat["gt_bb_coords"]  # (B, L, 4, 3)
        bb_coords_mask = gt_feat["bb_coords_mask"]  # (B, L, 4)

        gt_torsion_angles = gt_feat["gt_torsion_angles"]  # (B, L, 7, 2)
        torsion_angles_mask = gt_feat["torsion_angles_mask"]  # (B, L, 7)
        bb_coords_mask = bb_coords_mask * rigids_mask[..., None]

        # loss hyperparameters
        rot_loss_weight = self.cfg.rot_loss_weight
        rot_angle_loss_t_filter = self.cfg.rot_angle_loss_t_filter
        trans_loss_weight = self.cfg.trans_loss_weight
        bb_coords_loss_weight = self.cfg.bb_coords_loss_weight
        bb_coords_loss_t_filter = self.cfg.bb_coords_loss_t_filter
        bb_dist_map_loss_weight = self.cfg.bb_dist_map_loss_weight
        torsion_loss_weight = self.cfg.torsion_loss_weight
        bb_dist_map_loss_t_filter = self.cfg.bb_dist_map_loss_t_filter
        fape_loss_weight = self.cfg.fape_loss_weight

        # rotation loss
        gt_rot_angle = torch.norm(gt_rot_score, dim=-1, keepdim=True)  # (B, L, 1)
        gt_rot_axis = gt_rot_score / (gt_rot_angle + 1e-6)  # (B, L, 3)
        pred_rot_angle = torch.norm(pred_rot_score, dim=-1, keepdim=True)  # (B, L, 1)
        pred_rot_axis = pred_rot_score / (pred_rot_angle + 1e-6)  # (B, L, 3)
        # rotation axis loss
        rot_axis_mse = (gt_rot_axis - pred_rot_axis) ** 2 * rigids_mask[
            ..., None
        ]  # (B, L, 3)
        rot_axis_loss = torch.sum(rot_axis_mse, dim=(-2, -1)) / (
            rigids_mask.sum(dim=-1) + 1e-6
        )  # (B,)
        # rotation angle loss
        rot_angle_mse = (gt_rot_angle - pred_rot_angle) ** 2 * rigids_mask[
            ..., None
        ]  # (B, L, 1)
        rot_angle_loss = torch.sum(
            rot_angle_mse / gt_rot_score_norm[:, None, None] ** 2, dim=(-2, -1)
        ) / (
            rigids_mask.sum(dim=-1) + 1e-6
        )  # (B,)
        rot_angle_loss *= t > rot_angle_loss_t_filter
        rot_loss = rot_axis_loss + rot_angle_loss

        # translation loss
        trans_score_mse = (gt_trans_score - pred_trans_score) ** 2 * rigids_mask[
            ..., None
        ]
        trans_loss = torch.sum(
            trans_score_mse / gt_trans_score_norm[:, None, None] ** 2, dim=(-2, -1)
        ) / (
            rigids_mask.sum(dim=-1) + 1e-6
        )  # (B,)

        # auxiliary loss
        if bb_coords_loss_weight + bb_dist_map_loss_weight > 1e-4:
            # convert rigids to backbone atomic coordinates
            pred_atom37, pred_atom37_mask = atom14_to_atom37(pred_atom14, aatype)
            pred_bb_coords = pred_atom37[:, :, [0, 1, 2, 4], :]  # (B, L, 4, 3)

            # backbone coordinate loss
            bb_coords_mse = (gt_bb_coords - pred_bb_coords) ** 2 * bb_coords_mask[
                ..., None
            ]
            bb_coords_loss = torch.sum(
                bb_coords_mse, dim=(-3, -2, -1)  # (B, L, 4, 3)
            ) / (
                bb_coords_mask.sum(dim=(-2, -1)) + 1e-6
            )  # (B,)
            bb_coords_loss *= t < bb_coords_loss_t_filter  # (B,)

            # backbone distance map loss
            B = gt_bb_coords.size(0)
            gt_bb_dist_map = torch.cdist(
                gt_bb_coords.reshape(B, -1, 3), gt_bb_coords.reshape(B, -1, 3)
            )  # (B, L*4, L*4)
            bb_dist_map_mask = bb_coords_mask.reshape(
                B, 1, -1
            ) * bb_coords_mask.reshape(
                B, -1, 1
            )  # (B, L*4, L*4)
            bb_dist_map_mask *= gt_bb_dist_map < 6  # (B, L*4, L*4)
            pred_bb_dist_map = torch.cdist(
                pred_bb_coords.view(B, -1, 3), pred_bb_coords.view(B, -1, 3)
            )  # (B, L*4, L*4)
            bb_dist_map_mse = (
                gt_bb_dist_map - pred_bb_dist_map
            ) ** 2 * bb_dist_map_mask  # (B, L*4, L*4)
            bb_dist_map_loss = torch.sum(bb_dist_map_mse, dim=(-2, -1)) / (
                bb_dist_map_mask.sum(dim=(-2, -1)) + 1e-6
            )  # (B,)
            bb_dist_map_loss *= t < bb_dist_map_loss_t_filter  # (B,)
        else:
            bb_coords_loss = bb_dist_map_loss = torch.zeros_like(trans_loss)

        torsion_mse = (gt_torsion_angles - pred_torsions) ** 2 * torsion_angles_mask[
            ..., None
        ]
        torsion_loss = torsion_mse.sum(dim=(-1, -2, -3)) / torsion_angles_mask.sum(
            dim=(-1, -2)
        )

        if fape_loss_weight > 0:
            # backbone fape
            gt_frames = ru.Rigid.from_tensor_7(gt_feat["rigids_0"])
            bb_fape_loss = af2_loss.compute_fape(
                pred_frames=pred_rigids_0,
                target_frames=gt_frames,
                frames_mask=rigids_mask,
                pred_positions=pred_rigids_0.get_trans(),
                target_positions=gt_frames.get_trans(),
                positions_mask=rigids_mask,
                l1_clamp_distance=10.0,
                length_scale=10.0,
                eps=1e-4,
            )
            # sidechain fape
            gt_batch = {
                "atom14_gt_positions": gt_feat["atom14_gt_positions"],
                "atom14_alt_gt_positions": gt_feat["atom14_alt_gt_positions"],
                "atom14_atom_is_ambiguous": gt_feat["atom14_atom_is_ambiguous"],
                "atom14_gt_exists": gt_feat["atom14_gt_exists"],
                "atom14_alt_gt_exists": gt_feat["atom14_alt_gt_exists"],
                "atom14_atom_exists": gt_feat["atom14_atom_exists"],
            }
            renamed_gt_batch = af2_loss.compute_renamed_ground_truth(
                batch=gt_batch, atom14_pred_positions=pred_atom14
            )
            sc_fape_loss = af2_loss.sidechain_loss(
                sidechain_frames=pred_sidechain_frame[None],
                sidechain_atom_pos=pred_atom14[None],
                rigidgroups_gt_frames=gt_feat["rigidgroups_gt_frames"],
                rigidgroups_alt_gt_frames=gt_feat["rigidgroups_alt_gt_frames"],
                rigidgroups_gt_exists=gt_feat["rigidgroups_gt_exists"],
                renamed_atom14_gt_positions=renamed_gt_batch[
                    "renamed_atom14_gt_positions"
                ],
                renamed_atom14_gt_exists=renamed_gt_batch["renamed_atom14_gt_exists"],
                alt_naming_is_better=renamed_gt_batch["alt_naming_is_better"],
            )

            fape_loss = (
                bb_fape_loss + self.af2_config.loss.fape.sidechain.weight * sc_fape_loss
            )
        else:
            fape_loss = torch.zeros_like(trans_loss)

        # total loss
        loss = (
            rot_loss_weight * rot_loss
            + trans_loss_weight * trans_loss
            + bb_coords_loss_weight * bb_coords_loss
            + bb_dist_map_loss_weight * bb_dist_map_loss
            + torsion_loss_weight * torsion_loss
            + fape_loss_weight * fape_loss
        )  # (B,)

        # average across batch dim
        loss = loss.mean()
        rot_loss = rot_loss.mean()
        trans_loss = trans_loss.mean()
        bb_coords_loss = bb_coords_loss.mean()
        bb_dist_map_loss = bb_dist_map_loss.mean()
        torsion_loss = torsion_loss.mean()
        fape_loss = fape_loss.mean()

        # auxiliary info
        aux_info = {
            "total": loss.item(),
            "rot": rot_loss.item(),
            "trans": trans_loss.item(),
            "bb_coords": bb_coords_loss.item(),
            "bb_dist_map": bb_dist_map_loss.item(),
            "torsion": torsion_loss.item(),
            "fape": fape_loss.item(),
        }

        return loss, aux_info

    @torch.inference_mode()
    def reverse_sample(
        self,
        aatype,
        padding_mask,
        pretrained_node_repr=None,
        pretrained_edge_repr=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        assert not self.model_nn.training

        """ Reverse sampling. """
        rigids_mask = padding_mask.float()
        batch_size, seq_len = aatype.shape[:2]
        dt = torch.Tensor([1.0 / self.cfg.diffusion_steps] * batch_size).to(
            aatype.device
        )  # (B,)
        t = torch.Tensor([1.0] * batch_size).to(aatype.device)  # (B,)

        rigids_t = self.diffuser.sample_ref(
            n_samples=batch_size,
            seq_len=seq_len,
            device=aatype.device,
        )  # (B, L, 7)

        for step_t in tqdm(range(self.cfg.diffusion_steps), disable=False):

            model_out = self.model_nn(
                aatype=aatype,
                padding_mask=padding_mask,
                t=t,
                rigids_t=rigids_t,
                rigids_mask=rigids_mask,
                res_idx=None,
                pretrained_node_repr=pretrained_node_repr,
                pretrained_edge_repr=pretrained_edge_repr,
            )

            pred_rot_score = (
                self.diffuser.calc_rot_score(
                    rigids_t.get_rots(),
                    model_out["pred_rigids_0"].get_rots(),
                    t,
                )
                * rigids_mask[..., None]
            )

            pred_trans_score = (
                self.diffuser.calc_trans_score(
                    rigids_t.get_trans(),
                    model_out["pred_rigids_0"].get_trans(),
                    t[:, None, None],
                    use_torch=True,
                )
                * rigids_mask[..., None]
            )

            rigids_s = self.diffuser.reverse(
                rigids_t=rigids_t,  # (B, L)
                rot_score=pred_rot_score,  # (B, L, 3)
                trans_score=pred_trans_score,  # (B, L, 3)
                t=float(t[0]),
                dt=float(dt[0]),
            )  # (B, L, 7)

            # update rigids_t and t for the next step
            rigids_t = rigids_s  # (B, L, 7)
            t -= dt  # (B,)
            if t.min() < 0.01:
                break

        pred_atom37, atom37_mask = atom14_to_atom37(model_out["pred_atom14"], aatype)

        return {"atom37": pred_atom37, "atom37_mask": atom37_mask}
