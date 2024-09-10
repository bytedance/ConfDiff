# AlphaFold/OpenFold representation

## Step 1: generate MSA

We use a modified script from AlphaFlow (https://github.com/bjing2016/alphaflow/blob/master/scripts/mmseqs_query.py) to query the MSA from ColabFold's (https://github.com/sokrypton/ColabFold) server.

The `metadata.csv` file should contain two columns: `chain_name` the name of the protein and `seqres` the protein sequence.

```bash
python3 -m pretrain_repr.openfold.mmseqs_query_colabfold \
    /path/to/metadata.csv \
    --outdir /path/to/msa
```

## Step 2: download OpenFold ckpt

We use the weights `finetuning_no_templ_ptm_1.pt` (https://huggingface.co/nz/OpenFold) to generate sequence representation of OpenFold model. See https://github.com/aqlaboratory/openfold for details.


```bash
bash pretrain_repr/openfold/download_openfold_param.sh
```
The checkpoint will be downloaded at *pretrain_repr/openfold/openfold_params/finetuning_no_templ_ptm_1.pt*

## Step 3: generate node and edge repr

```bash
OUTPUT=/path/to/openfold_repr/
NUM_RECYCLES= # 0/3
# Example: generate node and edge representations with recycle=3
python3 -m pretrain_repr.openfold.make_openfold_repr \
        --input-csv-path /path/to/metadata.csv
        --msa-dir /path/to/msa \
        --output-dir $OUTPUT \ 
        --openfold-ckpt-fpath /path/to/ckpt \ # pretrain_repr/openfold/openfold_params/finetuning_no_templ_ptm_1.pt
        --num-recycles $NUM_RECYCLES
cat $OUTPUT/seqres_to_index.worker*.csv >> $OUTPUT/seqres_to_index.recycle$NUM_RECYCLES.csv
rm $OUTPUT/seqres_to_index.worker*.csv
```
- An index file `seqres_to_index.recycle0/3.csv` will be generated under `/path/to/open_repr/`


#### Multi-GPU script: we also provide a script to run inference with multiple GPUs in parallel:


```bash
#!/bin/bash
PREFIX=/path/to/confdiff/pretrain_repr/openfold/
NUM_GPU=8
INPUT_CSV_PATH=/path/to/metadata.csv
MSA_DIR=/path/to/msa
CKPT_PATH=/path/to/ckpt
NUM_RECYCLES=3
OUTPUT=/path/to/openfold_repr/

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
```