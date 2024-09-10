import numpy as np
import pandas as pd
from typing import Dict, Any
import torch
from src.utils import hydra_utils
from openfold.data import data_transforms
from openfold.np import residue_constants as rc
from openfold.utils import rigid_utils as ru
from src.data.full_atom.dataset import RCSBDataset
import os
import random
from Bio.PDB import PDBParser

pdb_parser = PDBParser(QUIET=True)
logger = hydra_utils.get_pylogger(__name__)


class GuidanceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: str,  # path to metadata csv file
        data_dir: str,
        diffuser=None,
        repr_loader=None,
        dynamic_batching: bool = False,
    ):
        if csv_path:
            self.df = pd.read_csv(csv_path, index_col=None)
        self.diffuser = diffuser
        self.data_dir = data_dir
        self.repr_loader = repr_loader
        self.csv_path = csv_path
        self.dynamic_batching = dynamic_batching

    def __len__(self):
        return len(self.df)

    def load_pdb(self, pdb_path):
        assert os.path.isfile(pdb_path), f"Cannot find pdb file at {pdb_path}."
        struct = pdb_parser.get_structure("", pdb_path)
        chain = struct[0][
            "A"
        ]  # each PDB file contains a single conformation, i.e., model 0
        seqres=""
        # load atomic coordinates
        seqlen =  len(list(chain.get_residues()))
        atom_coords = (
            np.zeros((seqlen, rc.atom_type_num, 3)) * np.nan
        )  # (seqlen, 37, 3)
        for residue in chain:
            # seq_idx = residue.id[1] - 1  # zero-based indexing
            if residue.has_id('CA'):  # get residues with CA atoms
                seqres += rc.restype_3to1[residue.resname.strip()]
            else:
                seqres += "X"
            for i,atom in enumerate(residue):
                if atom.name in rc.atom_order.keys():
                    atom_coords[i, rc.atom_order[atom.name]] = atom.coord

        return atom_coords,seqres

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        gt_energy_0 = row.normalized_energy

        gt_force_0 = np.load(
            f"{self.data_dir}/{row.chain_name}/{row.chain_name}_sample{row.sample_id}_opt_ca_force_clip.npy"
        )

        atom_coords,seqres = self.load_pdb(
            os.path.join(
                self.data_dir,
                f"{row.chain_name}",
                f"{row.chain_name}_sample{row.sample_id}_opt.pdb",
            )
        )
        aatype = torch.LongTensor([rc.restype_order_with_x[res] for res in seqres])

        atom_coords -= np.nanmean(atom_coords, axis=(0, 1), keepdims=True)
        all_atom_positions = torch.from_numpy(atom_coords)
        all_atom_mask = torch.all(~torch.isnan(all_atom_positions), dim=-1)

        all_atom_positions = torch.nan_to_num(
            all_atom_positions, 0.0
        )  # convert NaN to zero
        # ground truth backbone atomic coordinates
        gt_bb_coords = all_atom_positions[:, [0, 1, 2, 4], :]  # (seqlen, 4, 3)
        bb_coords_mask = all_atom_mask[:, [0, 1, 2, 4]]  # (seqlen, 4)

        openfold_feat_dict = {
            "aatype": aatype.long(),
            "all_atom_positions": all_atom_positions.double(),
            "all_atom_mask": all_atom_mask.double(),
        }

        openfold_feat_dict = data_transforms.atom37_to_frames(openfold_feat_dict)
        openfold_feat_dict = data_transforms.make_atom14_masks(openfold_feat_dict)
        openfold_feat_dict = data_transforms.make_atom14_positions(openfold_feat_dict)
        openfold_feat_dict = data_transforms.atom37_to_torsion_angles()(
            openfold_feat_dict
        )

        # ground truth rigids
        rigids_0 = ru.Rigid.from_tensor_4x4(
            openfold_feat_dict["rigidgroups_gt_frames"]
        )[:, 0]
        rigids_mask = openfold_feat_dict["rigidgroups_gt_exists"][:, 0]
        assert rigids_mask.sum() == torch.all(all_atom_mask[:, [0, 1, 2]], dim=-1).sum()
        t = max(0.01, random.random())

        diffused_feat_dict = self.diffuser.forward_marginal(
            rigids_0=rigids_0,
            t=t,
            diffuse_mask=rigids_mask.numpy(),
            as_tensor_7=False,
        )

        rigids_t = diffused_feat_dict["rigids_t"]

        for key, value in diffused_feat_dict.items():
            if isinstance(value, np.ndarray) or isinstance(value, np.float64):
                diffused_feat_dict[key] = torch.tensor(value)

        data_dict = {
            # 'seqres': seqres, # str
            "aatype": aatype.long(),
            "gt_energy_0": torch.tensor(gt_energy_0).float(),
            "gt_force_0": torch.tensor(gt_force_0).float(),
            "rigids_0": rigids_0.to_tensor_7().float(),  # (seqlen, 7)
            "rigids_t": rigids_t.to_tensor_7().float(),  # (seqlen, 7)
            "rigids_mask": rigids_mask.float(),  # (seqlen,)
            "t": torch.tensor(t).float(),  # (,)
            "rot_score": diffused_feat_dict["rot_score"].float(),  # (seqlen, 3)
            "trans_score": diffused_feat_dict["trans_score"].float(),  # (seqlen, 3)
            "rot_score_norm": diffused_feat_dict["rot_score_scaling"].float(),  # (,)
            "trans_score_norm": diffused_feat_dict[
                "trans_score_scaling"
            ].float(),  # (,)
            "gt_torsion_angles": openfold_feat_dict[
                "torsion_angles_sin_cos"
            ].float(),  # (seqlen,7,2)
            "torsion_angles_mask": openfold_feat_dict[
                "torsion_angles_mask"
            ].float(),  # (seqlen,7)
            "rigidgroups_gt_frames": openfold_feat_dict[
                "rigidgroups_gt_frames"
            ].float(),
            "rigidgroups_alt_gt_frames": openfold_feat_dict[
                "rigidgroups_alt_gt_frames"
            ].float(),
            "rigidgroups_gt_exists": openfold_feat_dict[
                "rigidgroups_gt_exists"
            ].float(),
            "atom14_gt_positions": openfold_feat_dict["atom14_gt_positions"].float(),
            "atom14_alt_gt_positions": openfold_feat_dict[
                "atom14_alt_gt_positions"
            ].float(),
            "atom14_atom_is_ambiguous": openfold_feat_dict[
                "atom14_atom_is_ambiguous"
            ].float(),
            "atom14_gt_exists": openfold_feat_dict["atom14_gt_exists"].float(),
            "atom14_alt_gt_exists": openfold_feat_dict["atom14_alt_gt_exists"].float(),
            "atom14_atom_exists": openfold_feat_dict["atom14_atom_exists"].float(),
            "gt_bb_coords": gt_bb_coords.float(),  # (seqlen, 4, 3)
            "bb_coords_mask": bb_coords_mask.float(),  # (seqlen, 4)
        }
        if self.repr_loader is not None:
            pretrained_repr = self.repr_loader.load(
                seqres=seqres
            )
            data_dict["pretrained_node_repr"] = pretrained_repr.get(
                "pretrained_node_repr", None
            )
            data_dict["pretrained_edge_repr"] = pretrained_repr.get(
                "pretrained_edge_repr", None
            )

        return data_dict

    def collate(self, batch_list):
        return RCSBDataset.collate(self, batch_list)
