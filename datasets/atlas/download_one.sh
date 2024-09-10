#!/bin/bash

# script: download one MD trajectory from ATLAS server
#    

function download_traj() {
    local pdb_id=$1     # chain_name of the trajectory to download
    local output_dir=$2 # directory to save the trajectory
    local traj_type=$3  # which type of traj to download. Options are 'protein', 'total', 'analysis'
    local par_dir=$output_dir/$pdb_id

    if [[ -z $pdb_id ]]; then 
        echo "Please provide pdb_id"
        exit 1
    fi

    if [[ ! -d $par_dir ]]; then
        mkdir -p $par_dir
    fi

    if [[ $traj_type == "protein" ]]; then
        local api_url="https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein"     # use this API to download all available protein conformation (10K * 3 replicates)
    elif [[ $traj_type == "total" ]]; then
        local api_url="https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/total"       # use this API to download all available frames for the full system (10K * 3 replicates)
    elif [[ $traj_type == "analysis" ]]; then
        local api_url="https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis"    # use this API to download a subset (1K * 3 replicates)
    else
        echo "Invalid data_type argument ($3). Please enter either 'protein', 'total', or 'analysis'." 
        return 1
    fi
    
    # -------------------- Check if already processed --------------------
    if [[ ( -f $par_dir/SUCCESS ) || ( -f $par_dir/README.txt && -f $par_dir/"$pdb_id"_prod_R3.xtc) ]]; then
        echo "$pdb_id exists - skip."
        touch $par_dir/SUCCESS
        return 0
    fi

    # -------------------- Download pipeline --------------------
    cd $par_dir
    local zip_dir=$par_dir/"$pdb_id".zip
    
    # Step 1: check if zip file is complete. Download if not
    if [[ ! -f $par_dir/DOWNLOAD_SUCCESS ]]; then

        # double check if download is success
        if [[ -f $zip_dir ]]; then
            if  unzip -tq $zip_dir ; then
                # success
                echo "$pdb_id download success"
                touch $par_dir/DOWNLOAD_SUCCESS
            else
                # download fail, remove current zip
                echo "ZIP file corrupted, removing $zip_dir ..."
            fi
        fi

        # download if not existed
        if [[ ! -f $par_dir/DOWNLOAD_SUCCESS ]]; then
            echo "Downloading $pdb_id to $zip_dir"
            curl -o $zip_dir -L "$api_url/$pdb_id" && touch $par_dir/DOWNLOAD_SUCCESS
        fi

    fi

    # Step 2: unzip the file, overwrite existing files
    unzip -o $zip_dir -d $par_dir && touch $par_dir/SUCCESS
    echo "$pdb_id done"

}

download_traj "$@"
