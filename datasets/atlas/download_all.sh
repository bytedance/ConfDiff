#!/bin/bash

# Make sure change directory to /atlas/ subfolder to call download_all.sh

chain_name=$1
output_root=$2 #  Your output path
max_proc=$3
data_type=protein

cat ./$chain_name | xargs -I {} -n 1 -P $max_proc bash ./download_one.sh {} $output_root $data_type
