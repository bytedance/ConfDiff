#!/bin/bash

DOWNLOAD_DIR="$1"
if [ -z "${DOWNLOAD_DIR}" ]; then
    echo "Error: please provide a download dir."
    exit 1
fi

mkdir --parents "${DOWNLOAD_DIR}"
rsync --recursive --links --perms --times --verbose --compress --info=progress2 --delete --port=33444 \
    rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ "${DOWNLOAD_DIR}"
