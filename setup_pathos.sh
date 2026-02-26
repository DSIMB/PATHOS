#!/bin/bash
# PATHOS Setup Script
# Downloads database files from Zenodo and sets up conda environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATABASE_DIR="${SCRIPT_DIR}/database"

echo "=== PATHOS Setup ==="

# 1. Create conda environment
echo "Creating conda environment..."
conda env create -f env/PATHOS_env.yml


# Function to download and verify file integrity
download_and_verify() {
    local url=$1
    local filename=$2
    local expected_md5=$3

    echo "Processing $filename..."
    
    # Check if file already exists and has correct MD5
    if [ -f "$filename" ]; then
        echo "File $filename already exists. Verifying integrity..."
        actual_md5=$(md5sum "$filename" | cut -d ' ' -f 1)
        if [ "$actual_md5" = "$expected_md5" ]; then
            echo "Integrity verified for $filename. Skipping download."
            return 0
        else
            echo "Integrity check failed for existing $filename. Re-downloading..."
        fi
    fi

    # Download file
    echo "Downloading $filename..."
    wget -c "$url" -O "$filename"

    # Verify integrity
    echo "Verifying integrity of $filename..."
    actual_md5=$(md5sum "$filename" | cut -d ' ' -f 1)
    if [ "$actual_md5" = "$expected_md5" ]; then
        echo "Integrity verified for $filename."
    else
        echo "ERROR: Integrity check failed for $filename. Expected $expected_md5, got $actual_md5"
        exit 1
    fi
}

# 2. Download and verify files individually
echo "Downloading database files from Zenodo..."

# af_index.sqlite (1.4 GB)
download_and_verify "https://zenodo.org/records/18140238/files/af_index.sqlite?download=1" "af_index.sqlite" "63a8e133b56ce22699076562975cff34"

# mmseqs_db.zip (2.3 GB)
download_and_verify "https://zenodo.org/records/18140238/files/mmseqs_db.zip?download=1" "mmseqs_db.zip" "823a681cc54a20b07efdf25245d641fb"

# MSAs.zip (1.0 GB)
download_and_verify "https://zenodo.org/records/18140238/files/MSAs.zip?download=1" "MSAs.zip" "b5c91e87441f2c98737cfb7f4132c97a"

# pathos.db (10.3 GB)
download_and_verify "https://zenodo.org/records/18140238/files/pathos.db?download=1" "pathos.db" "b0fac049b458b6c261c4f4f01ff958a3"

# 3. Extract zip archives
echo "Extracting archives into $DATABASE_DIR..."
if [ -f MSAs.zip ]; then
    unzip -o MSAs.zip -d "$DATABASE_DIR" && rm MSAs.zip
fi
if [ -f mmseqs_db.zip ]; then
    unzip -o mmseqs_db.zip -d "$DATABASE_DIR" && rm mmseqs_db.zip
fi
if [ -f database/fastas.zip ]; then
    unzip -o database/fastas.zip -d "$DATABASE_DIR" && rm database/fastas.zip
fi
if [ -f af_index.sqlite ]; then
    mv af_index.sqlite "$DATABASE_DIR/af_index.sqlite"
fi

cd ..

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with: conda activate PATHOS_env"
echo "Test with: python run_pathos.py --protein P51787 --mutation M1A"
