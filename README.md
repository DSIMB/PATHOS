# PATHOS - Protein variant Analysis Through Human-Optimized Scoring

PATHOS predicts pathogenicity of protein variants using protein language models (ESM-C 600M, Ankh2 Large, ESM2). Pre-computed scores for 139M+ variants across 17,574 human proteins.

## Installation

Set up PATHOS with a single script that installs dependencies and downloads the database.

**Prerequisites:** Linux, Conda, wget, unzip, ~35 GB disk space

```bash
git clone https://github.com/YOUR_USERNAME/PATHOS.git
cd PATHOS
./setup_pathos.sh
conda activate test_idris2
```


## Usage

Query pathogenicity scores for protein variants using UniProt IDs and mutation notation.

### Single mutation query

```bash
python run_pathos.py --protein P16501 --mutation M1A
```

### Batch query from file

```bash
python run_pathos.py --file variants.txt --output results.csv
```

### Filter results

```bash
python run_pathos.py --protein P16501 --min-score 0.9 --output pathogenic.csv
```

### Input file format

Supports TXT, TSV, and CSV formats. Headers are auto-detected and skipped.

**TXT/TSV (space or tab-separated):**

```
P16501 M1A R56V    # Multiple mutations per line
Q9Y6X3 M1C         # Single mutation
P10635             # Full scan (all 19 substitutions per position)
```

**CSV (comma-separated):**

```csv
Protein,Mutation
P16501,M1A
P16501,R56V
Q9Y6X3,M1C
```

## How it works

If all queried variants are already in the precomputed database (139M+ variants), results are returned instantly.

For variants not in the database, PATHOS performs de novo prediction:

1. Load UniProt sequences and validate mutations
2. Check/generate MSAs using mmseqs2 (if not already generated)
3. Compute PASTML conservation scores
4. Extract UniProt annotations and allele frequencies
5. Generate embeddings with ESMC 600M and Ankh2 Large
6. Run PATHOS inference (ensemble of both models)

## Output

Results are displayed in the terminal and exported to CSV with the following columns:

- UniProt ID
- Mutation (e.g., M1A)
- PATHOS score (0-1)
- Classification (Benign/Pathogenic)

## Score interpretation

PATHOS outputs a score between 0 and 1 indicating the probability of pathogenicity.

| Score | Classification |
|-------|----------------|
| < 0.63 | Benign |
| >= 0.63 | Pathogenic |

## Command-line options

Full list of available options for `run_pathos.py`.

| Option | Description |
|--------|-------------|
| `-i, --input` | Input file (TXT, TSV, or CSV) |
| `-o, --output` | Output CSV file |
| `--n-jobs` | Number of parallel workers for feature generation (default: 5). Increase for faster processing on multi-core systems. |
| `--batch-size` | Batch size for embedding generation (default: 100) |
| `--mmseqs-mem-limit` | Memory limit for mmseqs2 MSA generation (default: 8G) |
| `--batch-threshold` | Number of variants above which batched mode is enabled (default: 10000) |

## Citation

If you use PATHOS in your research, please cite:

Radjasandirane, R., Cretin, G., Diharce, J., de Brevern, A. G., & Gelly, J. C. (2025). PATHOS: Predicting Variant Pathogenicity by Combining Protein Language Models and Biological Features. medRxiv, 2025-12.


## Contact

radja.ragou@gmail.com
