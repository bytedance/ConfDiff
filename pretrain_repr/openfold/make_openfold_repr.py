#!/usr/bin/env python3
"""Make OpenFold pretrained representations

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================

from argparse import ArgumentParser

import random
import string
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from openfold.config import model_config
from openfold.data.data_pipeline import DataPipeline, make_sequence_features
from openfold.data.feature_pipeline import FeaturePipeline
from pretrain_repr.openfold.openfold_model import AlphaFold
from openfold.utils.import_weights import import_openfold_weights_

import torch

# =============================================================================
# Constants
# =============================================================================

ESMFOLD_CKPT = None  

# =============================================================================
# Functions
# =============================================================================


def canonize_chain_name(series):
    """Canonize all chain names to [pdb_id]_[chain_id]"""
    return series.str.replace(".pdb", "").str.replace(".", "_")


# =============================================================================
# Main
# =============================================================================


def main(args):
    """
    Featurizes amino acids into node and edge embeddings.
    Embeddings are stored as two separate npy files:
        [chain_name].lm_node_repr.npy
        [chain_name].trunk_node_repr.recycle[recycle_num].npy
        [chain_name].trunk_edge_repr.recycle[recycle_num].npy
    """
    # load data

    df = pd.read_csv(args.input_csv_path)
    chain_name_col = "chain_name" if "chain_name" in df.columns else "name"
    seqres_col = "seqres"
    df[chain_name_col] = canonize_chain_name(df[chain_name_col])
    chain_df = df[[chain_name_col, seqres_col]]

    # check output dir, skip existed
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    index_file = output_root.joinpath(f"seqres_to_index.recycle{args.num_recycles}.csv")

    if index_file.exists():
        existed = pd.read_csv(index_file)
        chain_df_existed = chain_df[seqres_col].isin(existed["seqres"])
        if args.worker_id == 0:
            print(
                f"\t{chain_df_existed.sum():,}/{len(chain_df_existed):,} (recycle {args.num_recycles}) embeddings already existed. Skip them for generation"
            )
        chain_df = chain_df[~chain_df_existed]
        existed = chain_df_existed.sum()
    else:
        if args.worker_id == 0:
            with open(index_file, "w") as handle:
                handle.write("seqres,index\n")
        existed = 0
    chain_df = chain_df.drop_duplicates(subset=seqres_col)

    # -------------------- Generate Openfold repr --------------------
    chain_df = chain_df.iloc[args.worker_id :: args.num_workers]
    print(
        f"[Worker {args.worker_id}/{args.num_workers}] generating embeddings for {len(chain_df)} chains"
    )
    # load OpenFold model
    af2_config = model_config(
        "model_3_ptm", 
        train=False, 
        low_prec=False
    )
    af2_config.data.common.max_recycling_iters = args.num_recycles
    model = AlphaFold(af2_config).to('cuda:0')
    model = model.eval()
    d = torch.load(args.openfold_ckpt_fpath)
    if "ema" in d:
        d = d["ema"]["params"]
    import_openfold_weights_(model=model, state_dict=d)

    data_pipeline = DataPipeline(template_featurizer=None)
    feature_pipeline = FeaturePipeline(af2_config.data) 
    
    failed = 0
    oom = 0
    newly_done = 0

    index_f = open(output_root / f"seqres_to_index.worker{args.worker_id}.csv", "w")

    for idx in tqdm(range(len(chain_df)), total=len(chain_df)):
        row = chain_df.iloc[idx]
        seqres = row.seqres
        chain_name = row.chain_name

        mmcif_feats = make_sequence_features(
                    sequence=seqres,
                    description=chain_name,
                    num_res=len(seqres),
                )
        msa_features = data_pipeline._process_msa_feats(f'{args.msa_dir}/{chain_name}/a3m', seqres, alignment_index=None)
        processed_feature_dict = feature_pipeline.process_features({**mmcif_feats, **msa_features}, mode='predict')
        processed_feature_dict = {
            k:torch.as_tensor(v, device='cuda:0') 
            for k,v in processed_feature_dict.items()
        }

        
        try:
            with torch.no_grad():
                out = model(processed_feature_dict)
                node_repr, edge_repr = out['evo_single'], out['evo_pair']
        except Exception as e:
            if "out of memory" in str(e):
                print(
                    f"[Worker {args.worker_id}/{args.num_workers}] CUDA OOM, skipping batch {idx}: {chain_name}",
                    flush=True,
                )
                torch.cuda.empty_cache()
                oom += 1
                continue
            print(
                f"[Worker {args.worker_id}/{args.num_workers}] error processing batch {idx} ({chain_name}): {e}",
                flush=True,
            )
            failed += 1
            continue

        node_repr = node_repr.cpu().numpy()
        edge_repr = edge_repr.cpu().numpy()

        output_root.joinpath(chain_name[:2]).mkdir(exist_ok=True)
        assert node_repr.shape[0] == len(
            seqres
        ), f"{chain_name}: length mismatch: {node_repr.shape[0]} vs {len(seqres)}"

        prefix = str(output_root / chain_name[:2] / chain_name)
        if Path(f"{prefix}.node_repr.npy").exists():
            # get a new name
            suffix = ""
            while Path(f"{prefix}{suffix}.lm_node_repr.npy").exists():
                suffix = (
                    "_"
                    + "".join(
                        random.choice(string.hexdigits) for _ in range(2)
                    ).lower()
                )

            prefix = str(output_root / chain_name[:2] / f"{chain_name}{suffix}")
            chain_name = f"{chain_name}{suffix}"

        if Path(f"{prefix}.node_repr.npy").exists():
            raise ValueError(f"{prefix}.lm_node_repr.npy already exists")

        np.save(
            f"{prefix}.node_repr.recycle{args.num_recycles}.npy",
            node_repr,
        )
        np.save(
            f"{prefix}.edge_repr.recycle{args.num_recycles}.npy",
            edge_repr,
        )
        index_f.write(f"{seqres},{chain_name[:2]}/{chain_name}\n")
        newly_done += 1

    print(
        f"[Worker {args.worker_id}/{args.num_workers}] already done: {existed}, success: {newly_done} failed: {failed} (OOM: {oom})"
    )
    index_f.close()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--input-csv-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--msa-dir", type=str, required=True)
    parser.add_argument("--openfold-ckpt-fpath", default='pretrain_repr/openfold/openfold_params/finetuning_no_templ_1.pt')
    parser.add_argument(
        "--num-recycles", type=int, default=3
    )  # recycle=0: single forward pass, recycle=3 used in OpenFold training
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)

    args, _ = parser.parse_known_args()

    main(args)
