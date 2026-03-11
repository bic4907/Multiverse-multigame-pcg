#!/bin/bash

# Dataset Setup Script
# Reassembles and extracts the dataset from split archive files.
#
# The dataset contains:
#   - VGLC (Video Game Level Corpus) dataset
#   - Human-authored levels from the PCGRL environment
#
# Usage: bash setup_dataset.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Reassembling dataset archive..."
cat dataset/dataset.tar.xz.* > dataset/dataset.tar.xz

echo "Extracting dataset..."
tar -xJf dataset/dataset.tar.xz -C .

echo "Cleaning up temporary archive..."
rm dataset/dataset.tar.xz

echo "Done! Dataset is ready at ./dataset/"

