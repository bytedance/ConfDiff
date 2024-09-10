#!/usr/bin/env python3
"""Make ESM embedding from LM or trunk layers

----------------
[License]
SPDX-License-Identifier: Apache-2.0
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

import torch

from .esmfold import ESMFold

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

    # -------------------- Generate ESM repr --------------------
    chain_df = chain_df.iloc[args.worker_id :: args.num_workers]
    print(
        f"[Worker {args.worker_id}/{args.num_workers}] generating embeddings for {len(chain_df)} chains"
    )
    # load ESMFold model
    esm_enc = ESMFold(ckpt_fpath=args.esm_ckpt_fpath).to("cuda:0")
    assert esm_enc is not None, "Load ESM model fail"

    failed = 0
    oom = 0
    newly_done = 0

    if len(chain_df) % args.batch_size == 0:
        total_batches = len(chain_df) // args.batch_size
    else:
        total_batches = len(chain_df) // args.batch_size + 1

    index_f = open(output_root / f"seqres_to_index.worker{args.worker_id}.csv", "w")

    for batch_ix in tqdm(range(total_batches), total=total_batches):
        batch_df = chain_df[
            batch_ix * args.batch_size : (batch_ix + 1) * args.batch_size
        ]
        if len(batch_df) == 0:
            break
        seqres_batch = list(batch_df[seqres_col])
        chain_name_batch = list(batch_df[chain_name_col])

        try:
            lm_node_repr, trunk_node_repr, trunk_edge_repr, mask = esm_enc.infer(
                sequences=seqres_batch, num_recycles=args.num_recycles
            )
        except Exception as e:
            if "out of memory" in str(e):
                print(
                    f"[Worker {args.worker_id}/{args.num_workers}] CUDA OOM, skipping batch {batch_ix}: {', '.join(chain_name_batch)}",
                    flush=True,
                )
                torch.cuda.empty_cache()
                oom += 1
                continue
            print(
                f"[Worker {args.worker_id}/{args.num_workers}] error processing batch {batch_ix} ({', '.join(chain_name_batch)}): {e}",
                flush=True,
            )
            failed += 1
            continue
        lm_node_repr = lm_node_repr.cpu().numpy()
        trunk_node_repr = trunk_node_repr.cpu().numpy()
        trunk_edge_repr = trunk_edge_repr.cpu().numpy()
        mask = mask.cpu().numpy().astype(bool)

        for ix, (chain_name, seqres) in enumerate(zip(chain_name_batch, seqres_batch)):
            # make subdir
            output_root.joinpath(chain_name[:2]).mkdir(exist_ok=True)
            lm_node_repr_ = lm_node_repr[ix, mask[ix], ...]
            trunk_node_repr_ = trunk_node_repr[ix, mask[ix], ...]
            trunk_edge_repr_ = trunk_edge_repr[ix, mask[ix], ...]
            assert lm_node_repr_.shape[0] == len(
                seqres
            ), f"{chain_name}: length mismatch: {lm_node_repr.shape[0]} vs {len(seqres)}"

            prefix = str(output_root / chain_name[:2] / chain_name)
            if Path(f"{prefix}.lm_node_repr.npy").exists():
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

            if Path(f"{prefix}.lm_node_repr.npy").exists():
                raise ValueError(f"{prefix}.lm_node_repr.npy already exists")

            np.save(f"{prefix}.lm_node_repr.npy", lm_node_repr_)
            np.save(
                f"{prefix}.trunk_node_repr.recycle{args.num_recycles}.npy",
                trunk_node_repr_,
            )
            np.save(
                f"{prefix}.trunk_edge_repr.recycle{args.num_recycles}.npy",
                trunk_edge_repr_,
            )
            index_f.write(f"{seqres},{chain_name[:2]}/{chain_name}\n")
        newly_done += args.batch_size

    print(
        f"[Worker {args.worker_id}/{args.num_workers}] already done: {existed}, success: {newly_done} failed: {failed} (OOM: {oom})"
    )
    index_f.close()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--input-csv-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--esm-ckpt-fpath", default=None)
    parser.add_argument(
        "--num-recycles", type=int, default=3
    )  # 1 fwd pass + 3 recycles == esm's max recycle number 4
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)

    args, _ = parser.parse_known_args()

    main(args)
