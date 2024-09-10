#!/bin/bash
# Scripts to run ESMFold representation generation using multiple GPUs

# ************************************************************************
# [Change arugments
# ************************************************************************
PREFIX=ConfDiff/pretrain_repr/openfold/
NUM_GPU=8
INPUT_CSV_PATH=
MSA_DIR=
CKPT_PATH=ConfDiff/pretrain_repr/openfold/openfold_params/finetuning_no_templ_1.pt
NUM_RECYCLES=3
OUTPUT=



for GPU_ID in $(seq 0 $(($NUM_GPU-1))); do
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 $PREFIX/make_openfold_repr.py \
        --input-csv-path $INPUT_CSV_PATH \
        --msa-dir $MSA_DIR \
        --output-dir $OUTPUT \
        --openfold-ckpt-fpath $CKPT_PATH \
        --num-recycles $NUM_RECYCLES \
        --num-workers $NUM_GPU \
        --worker-id $GPU_ID &
done

wait

# concat all index files
cat $OUTPUT/seqres_to_index.worker*.csv >> $OUTPUT/seqres_to_index.recycle$NUM_RECYCLES.csv
rm $OUTPUT/seqres_to_index.worker*.csv