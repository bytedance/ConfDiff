"""Loader for sequence representations

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""
import os
import torch
import numpy as np
import pandas as pd
from pathlib import PosixPath, Path
from typing import Dict, Union
from src.utils.hydra_utils import get_pylogger

logger = get_pylogger(__name__)

PATH_TYPE = Union[str, PosixPath]


class OpenFoldReprLoader(object):
    def __init__(
        self,
        data_root: PATH_TYPE,
        num_recycles: int = 3,
        node_size: int = 384,
        edge_size: int = 128,
    ):
        assert os.path.exists(data_root), f"pretrain_repr_root not found: {data_root}"
        self.data_root = Path(data_root)
        self.num_recycles = num_recycles
        self.node_size = node_size
        self.edge_size = edge_size
        csv_path = self.data_root / f"seqres_to_index.recycle{num_recycles}.csv"
        assert csv_path.is_file(), f"Cannot find metadata csv at {csv_path}"
        self.metadata = pd.read_csv(csv_path, index_col="seqres")
        assert self.metadata.index.nunique() == len(self.metadata)

    def load(self, seqres: str) -> Dict[str, torch.Tensor]:
        """Load node and/or edge representations from pretrained model
        Returns:
            {
                node_repr: Tensor[seqlen, repr_dim], float
                edge_repr: Tensor[seqlen, seqlen, repr_dim], float
            }
        """

        repr_dict = {}

        lookup = self.metadata.loc[seqres]
        assert len(lookup) == 1
        prefix = lookup.values[0]
        # -------------------- Node repr --------------------
        if self.node_size > 0:
            node_repr_path = (
                self.data_root
                / f"{prefix}"
                / f"{prefix}_recycle{self.num_recycles:d}_single_repr.npy"
            )
            if os.path.exists(node_repr_path):
                node_repr = np.load(node_repr_path)
                repr_dict["pretrained_node_repr"] = torch.from_numpy(node_repr).float()
            else:
                raise FileNotFoundError(
                    f"{prefix}: node_repr not found: {str(node_repr_path)}"
                )
        # -------------------- Edge repr --------------------
        if self.edge_size > 0:
            edge_repr_path = (
                self.data_root
                / f"{prefix}"
                / f"{prefix}_recycle{self.num_recycles:d}_pair_repr.npy"
            )
            if os.path.exists(edge_repr_path):
                edge_repr = np.load(edge_repr_path)
                repr_dict["pretrained_edge_repr"] = torch.from_numpy(edge_repr).float()
            else:
                raise FileNotFoundError(
                    f"{prefix}: edge_repr not found: {str(edge_repr_path)}"
                )
        return repr_dict


class ESMFoldReprLoader(object):
    def __init__(
        self,
        data_root: PATH_TYPE,
        num_recycles: int = 3,
        node_size: int = 1024,
        edge_size: int = 128,
    ):
        assert os.path.exists(data_root), f"pretrain_repr_root not found: {data_root}"
        self.data_root = Path(data_root)
        self.num_recycles = num_recycles
        self.node_size = node_size
        self.edge_size = edge_size
        csv_path = (
            self.data_root
            / f"recycle{num_recycles:d}"
            / f"seqres_to_index.recycle{num_recycles:d}.csv"
        )
        assert csv_path.is_file(), f"Cannot find metadata csv at {csv_path}"
        self.metadata = pd.read_csv(csv_path, index_col="seqres")
        assert self.metadata.index.nunique() == len(self.metadata)

    def load(self, seqres: str) -> Dict[str, torch.Tensor]:
        """Load node and/or edge representations from pretrained model
        Returns:
            {
                node_repr: Tensor[seqlen, repr_dim], float
                edge_repr: Tensor[seqlen, seqlen, repr_dim], float
            }
        """

        repr_dict = {}

        lookup = self.metadata.loc[seqres]
        assert len(lookup) == 1
        prefix = lookup.values[0]
        # -------------------- Node repr --------------------
        if self.node_size > 0:
            node_repr_path = (
                self.data_root
                / f"recycle{self.num_recycles:d}"
                / f"{prefix}.trunk_node_repr.recycle{self.num_recycles:d}.npy"
            )
            if os.path.exists(node_repr_path):
                node_repr = np.load(node_repr_path)
                repr_dict["pretrained_node_repr"] = torch.from_numpy(node_repr).float()
            else:
                raise FileNotFoundError(
                    f"{prefix}: node_repr not found: {str(node_repr_path)}"
                )
        # -------------------- Edge repr --------------------
        if self.edge_size > 0:
            edge_repr_path = (
                self.data_root
                / f"recycle{self.num_recycles:d}"
                / f"{prefix}.trunk_edge_repr.recycle{self.num_recycles:d}.npy"
            )
            if os.path.exists(edge_repr_path):
                edge_repr = np.load(edge_repr_path)
                repr_dict["pretrained_edge_repr"] = torch.from_numpy(edge_repr).float()
            else:
                raise FileNotFoundError(
                    f"{prefix}: edge_repr not found: {str(edge_repr_path)}"
                )
        return repr_dict
