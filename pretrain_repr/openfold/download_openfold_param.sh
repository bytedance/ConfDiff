# sudo apt-get install git-lfs
git lfs install

set -e

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

# URL="https://huggingface.co/nz/OpenFold/finetuning_no_templ_ptm_1.pt"

DOWNLOAD_DIR="${1}/openfold_params/"
mkdir -p "${DOWNLOAD_DIR}"
# cd "${DOWNLOAD_DIR}"
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/nz/OpenFold.git ${DOWNLOAD_DIR}
cd "${DOWNLOAD_DIR}"
# cd "OpenFold"
git lfs pull --include="finetuning_no_templ_ptm_1.pt"
# cd ../
rm -rf ".git"
