#!/bin/bash
# PATHOS Setup Script
# Downloads database files from Zenodo and sets up conda environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATABASE_DIR="${SCRIPT_DIR}/database"

ZENODO_ARCHIVE_URL="https://zenodo.org/api/records/18140238/files-archive"

echo "=== PATHOS Setup ==="

# 1. Create conda environment
echo "Creating conda environment..."
conda env create -f env/env_idris.yml

# 2. Change to database directory
cd "$DATABASE_DIR"

# 3. Download entire Zenodo archive
echo "Downloading Zenodo archive..."
wget -c "$ZENODO_ARCHIVE_URL" -O zenodo_archive.zip

# 4. Extract archive
echo "Extracting archive..."
unzip -o zenodo_archive.zip && rm zenodo_archive.zip

# 5. Extract nested archives (MSAs.zip, mmseqs_db.zip, uniprot.zip)
if [ -f MSAs.zip ]; then
    unzip -o MSAs.zip && rm MSAs.zip
fi
if [ -f mmseqs_db.zip ]; then
    unzip -o mmseqs_db.zip && rm mmseqs_db.zip
fi
if [ -f uniprot.zip ]; then
    unzip -o uniprot.zip && rm uniprot.zip
fi

cd ..

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with: conda activate test_idris2"
echo "Test with: python run_pathos.py -i example_input.txt -o test_output.csv"