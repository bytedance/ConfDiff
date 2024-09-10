import torch
from torch import nn
from src.models.full_atom.utils.all_atom import atom14_to_atom37
from openfold.utils import rigid_utils as ru
from tqdm import tqdm
from src.models.full_atom.score_network import BaseScoreNetwork
from torch import einsum


class GuidanceScoreNetwork(BaseScoreNetwork):
    def __init__(
        self,
        cond_model_nn,
        cond_ckpt_path,
        cfg,
        uncond_model_nn=None,
        uncond_ckpt_path=None,
        **kwargs
    ):
        super(GuidanceScoreNetwork, self).__init__(
            cond_model_nn,
            cfg,
        )
        self.diffuser = None
        if cond_ckpt_path:
            cond_ckpt = torch.load(cond_ckpt_path, map_location="cpu")["state_dict"]
            cond_state_dict = {}
            for key in cond_ckpt.keys():
                if key.startswith("score_network.model_nn."):
                    cond_state_dict[key[len("score_network.model_nn.") :]] = cond_ckpt[key]

            self.model_nn.load_state_dict(cond_state_dict, strict=False)
            del cond_ckpt
        for param in self.model_nn.parameters():
            param.requires_grad = False

        self.uncond_model_nn = uncond_model_nn

        if uncond_ckpt_path:
            uncond_ckpt = torch.load(uncond_ckpt_path, map_location="cpu")["state_dict"]
            uncond_state_dict = {}
            for key in uncond_ckpt.keys():
                if key.startswith("score_network.model_nn"):
                    uncond_state_dict[key[len("score_network.model_nn.") :]] = (
                        uncond_ckpt[key]
                    )

            self.uncond_model_nn.load_state_dict(uncond_state_dict, strict=False)
            del uncond_ckpt

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
        self.diffuser._so3_diffuser.use_cached_score = False
        rigids_t = ru.Rigid.from_tensor_7(rigids_t)
        rigids_mask = rigids_mask * padding_mask

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

        input_feat = {
            "aatype": aatype,
            "t": t,
            "rigids_t": rigids_t,
            "rigids_mask": rigids_mask,
            "padding_mask": padding_mask,
            "gt_feat": gt_feat,
            "res_idx": res_idx,
            "pretrained_node_repr": pretrained_node_repr,
            "pretrained_edge_repr": pretrained_edge_repr,
        }

        loss = self.loss_fn(input_feat, model_out)

        return loss, {"total": loss.item()}

    def loss_fn(
        self,
        input_feat,
        model_out,
        **kwargs,
    ):

        return torch.tensor(0)

    def forward_guidance_model(
        self,
        input_feat,
    ):
        model_out = self.model_nn(**input_feat)

        return 0, model_out

    def reverse_sample(
        self,
        aatype,
        padding_mask,
        pretrained_node_repr=None,
        pretrained_edge_repr=None,
        **kwargs,
    ) -> None:
        assert not self.model_nn.training
        self.diffuser._so3_diffuser.use_cached_score = True

        """ Reverse sampling. """
        rigids_mask = padding_mask.float()

        batch_size, seq_len = aatype.shape[:2]
        dt = torch.Tensor([1.0 / self.cfg.diffusion_steps] * batch_size).to(
            aatype.device
        )  # (B,)
        t = torch.Tensor([1.0] * batch_size).to(aatype.device)  # (B,)

        rigids_t = self.diffuser.sample_ref(
            n_samples=batch_size, seq_len=seq_len, device=aatype.device
        )  # (B, L, 7)
        for step_t in tqdm(range(self.cfg.diffusion_steps)):

            uncond_model_out = None

            if self.uncond_model_nn and self.cfg.clsfree_guidance_strength < 1.0:
                uncond_model_out = self.uncond_model_nn(
                    aatype=aatype,
                    padding_mask=padding_mask,
                    t=t,
                    rigids_t=rigids_t,
                    rigids_mask=rigids_mask,
                    res_idx=None,
                    pretrained_node_repr=None,
                    pretrained_edge_repr=None,
                )

                uncond_pred_rot_score = (
                    self.diffuser.calc_rot_score(
                        rigids_t.get_rots(),
                        uncond_model_out["pred_rigids_0"].get_rots(),
                        t,
                    )
                    * rigids_mask[..., None]
                )

                uncond_pred_trans_score = (
                    self.diffuser.calc_trans_score(
                        rigids_t.get_trans(),
                        uncond_model_out["pred_rigids_0"].get_trans(),
                        t[:, None, None],
                        use_torch=True,
                    )
                    * rigids_mask[..., None]
                )
            else:
                uncond_pred_rot_score, uncond_pred_trans_score = 0, 0

            input_feat = {
                "aatype": aatype,
                "t": t,
                "rigids_t": rigids_t,
                "rigids_mask": rigids_mask,
                "padding_mask": padding_mask,
                "pretrained_node_repr": pretrained_node_repr,
                "pretrained_edge_repr": pretrained_edge_repr,
            }

            pred_force_t, model_out = self.forward_guidance_model(
                input_feat=input_feat,
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

            pred_rot_score = (
                self.cfg.clsfree_guidance_strength * pred_rot_score
                + (1 - self.cfg.clsfree_guidance_strength) * uncond_pred_rot_score
            )
            pred_trans_score = (
                self.cfg.clsfree_guidance_strength * pred_trans_score
                + (1 - self.cfg.clsfree_guidance_strength) * uncond_pred_trans_score
            )

            pred_trans_score = (
                pred_trans_score - self.cfg.force_guidance_strength * pred_force_t
            )

            rigids_s = self.diffuser.reverse(
                rigids_t=rigids_t,  # (B, L)
                rot_score=pred_rot_score,  # (B, L, 3)
                trans_score=pred_trans_score,  # (B, L, 3)
                t=float(t[0]),
                dt=float(dt[0]),
            )  # (B, L, 7)

            rigids_t = rigids_s  # (B, L, 7)
            t -= dt  # (B,)
            if t.min() < 0.01:
                break
        pred_atom37, atom37_mask = atom14_to_atom37(model_out["pred_atom14"], aatype)
        return {"atom37": pred_atom37, "atom37_mask": atom37_mask}


class ForceGuidance(GuidanceScoreNetwork):
    def __init__(
        self,
        cond_model_nn,
        cond_ckpt_path,
        cfg,
        uncond_model_nn=None,
        uncond_ckpt_path=None,
    ):
        super(ForceGuidance, self).__init__(
            cond_model_nn,
            cond_ckpt_path,
            cfg,
            uncond_model_nn,
            uncond_ckpt_path,
        )
        for param in self.model_nn.structure_module.pred_force_t_net.parameters():
            param.requires_grad = True
        if self.model_nn.structure_module.pred_force_0:
            for param in self.model_nn.structure_module.pred_force_0_net.parameters():
                param.requires_grad = True

    def loss_fn(
        self,
        input_feat,
        model_out,
        **kwargs,
    ):

        pred_force_0 = model_out["pred_force_0"]
        pred_force_t = model_out["pred_force_t"]

        gt_feat = input_feat["gt_feat"]
        trans_0 = gt_feat["rigids_0"][..., 4:]
        trans_t = input_feat["rigids_t"].get_trans()
        t = input_feat["t"]
        rigids_mask = input_feat["rigids_mask"]

        pred_trans_score = (
            self.diffuser.calc_trans_score(
                trans_t,
                model_out["pred_rigids_0"].get_trans(),
                t[:, None, None],
                use_torch=True,
            )
            * rigids_mask[..., None]
        )

        energy_weight = torch.exp(-gt_feat["gt_energy_0"])
        gt_trans_score = gt_feat["trans_score"]

        sigma_t = 1.0 / gt_feat["trans_score_norm"]
        seqlen = gt_trans_score.shape[1]

        mu_t = einsum(
            "a,blc->ablc", torch.sqrt(1.0 - sigma_t**2), trans_0
        )  # (B, B, L, 3)
        q_xt_x0 = torch.exp(
            -0.5 * (gt_trans_score**2).sum(dim=(-1, -2)) * sigma_t**2 / seqlen
        )  # (B,)
        zeta = (
            pred_trans_score[:, None, ...]
            + (trans_t[:, None, ...] - mu_t) * (1.0 / sigma_t**2)[:, None, None, None]
        )  # (B, B, L, 3)

        numerator = einsum(
            "a, b, abcd->abcd", q_xt_x0, energy_weight, zeta
        )  # (B, B, L, 3)
        denominator = q_xt_x0 * energy_weight.sum()  # (B,)
        gt_force_t = numerator.sum(dim=1) / denominator[:, None, None]  # (B, L, 3)

        # clamp norm
        gt_force_t_norm = torch.linalg.norm(gt_force_t, dim=-1)
        gt_force_t = (
            gt_force_t
            * (torch.clip(gt_force_t_norm, 0, 100) / (gt_force_t_norm + 1e-8))[
                ..., None
            ]
        )

        # force matching
        pred_force_t = (1.0 - t)[..., None, None] * pred_force_0 + ((1.0 - t) * t)[
            ..., None, None
        ] * pred_force_t
        force_score_mse = (
            (pred_force_t - 0.1 * gt_force_t) ** 2
            + (pred_force_0 - gt_feat["gt_force_0"]) ** 2
        ) * rigids_mask[..., None]
        force_loss = torch.sum(force_score_mse, dim=(-2, -1)) / (
            rigids_mask.sum(dim=-1) + 1e-6
        )

        return force_loss.mean()

    def forward_guidance_model(
        self,
        input_feat,
    ):

        model_out = self.model_nn(**input_feat)

        t = input_feat["t"]
        pred_force_t = torch.nn.functional.normalize(
            model_out["pred_force_t"], dim=-1, p=2
        )
        pred_force_0 = torch.nn.functional.normalize(
            model_out["pred_force_0"], dim=-1, p=2
        )
        pred_force_t = (1.0 - t)[..., None, None] * pred_force_0 + ((1.0 - t) * t)[
            ..., None, None
        ] * pred_force_t

        return pred_force_t, model_out


class EnergyGuidance(GuidanceScoreNetwork):
    def __init__(
        self,
        cond_model_nn,
        cond_ckpt_path,
        cfg,
        uncond_model_nn=None,
        uncond_ckpt_path=None,
        **kwargs
    ):
        super(EnergyGuidance, self).__init__(
            cond_model_nn,
            cond_ckpt_path,
            cfg,
            uncond_model_nn,
            uncond_ckpt_path,
        )
        for param in self.model_nn.structure_module.pred_energy_t_net.parameters():
            param.requires_grad = True
        for param in self.model_nn.structure_module.energy_t_final.parameters():
            param.requires_grad = True

    def loss_fn(
        self,
        input_feat,
        model_out,
        **kwargs,
    ):
        gt_energy_0 = input_feat["gt_feat"]["gt_energy_0"]
        pred_energy_t = model_out["pred_energy_t"]
        loss = -torch.exp(-gt_energy_0)[..., None] * nn.functional.log_softmax(
            -pred_energy_t, dim=0
        )
        return loss.mean()

    def forward_guidance_model(
        self,
        input_feat,
    ):
        with torch.enable_grad():
            input_feat["rigids_t"] = (
                input_feat["rigids_t"].to_tensor_7().requires_grad_()
            )
            model_out = self.model_nn(**input_feat)
            pred_force_t = torch.autograd.grad(
                model_out["pred_energy_t"].sum(), input_feat["rigids_t"]
            )[0]
        pred_force_t = torch.nn.functional.normalize(pred_force_t[..., 4:], dim=-1, p=2)

        return pred_force_t, model_out