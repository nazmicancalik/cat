#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DOWNLOAD_DIR="../data/scannet"
echo "ScanNet v2 will be downloaded to" $SCRIPT_DIR/$DOWNLOAD_DIR
python2 "$SCRIPT_DIR/download-scannet.py" -o "$SCRIPT_DIR/$DOWNLOAD_DIR" --type _vh_clean_2.ply _vh_clean_2.labels.ply