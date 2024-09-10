#!/bin/bash
# Scripts to run ESMFold representation generation using multiple GPUs

# ************************************************************************
# [Change arugments
# ************************************************************************
PREFIX=
NUM_GPU=
INPUT_CSV_PATH=
CKPT_PATH=
NUM_RECYCLES=0
OUTPUT=
BATCH_SIZE=1

for GPU_ID in $(seq 0 $(($NUM_GPU-1))); do
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 $PREFIX/make_esm_repr.py \
        --input-csv-path $INPUT_CSV_PATH \
        --output-dir $OUTPUT \
        --esm-ckpt-fpath $CKPT_PATH \
        --num-recycles $NUM_RECYCLES \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_GPU \
        --worker-id $GPU_ID &
done

wait

# concat all index files
cat $OUTPUT/seqres_to_index.worker*.csv >> $OUTPUT/seqres_to_index.recycle$NUM_RECYCLES.csv
rm $OUTPUT/seqres_to_index.worker*.csv