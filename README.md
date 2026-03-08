# PATHOS - Predicting variant pathogenicity by combining Protein Language Models and biological features

PATHOS predicts pathogenicity of protein variants using protein language models (ESM-C 600M, Ankh2 Large). Pre-computed scores for 140M+ variants across 17,734 human proteins.

## Installation

Set up PATHOS with a single script that installs dependencies and downloads the database.

**Prerequisites:** ~35 GB disk space

```bash
git clone https://github.com/DSIMB/PATHOS.git
cd PATHOS
./setup_pathos.sh
conda activate PATHOS_env
```


## Quick start

Query pathogenicity scores for protein variants using UniProt IDs and mutation notation.

### Single mutation query

```bash
python run_pathos.py --protein P51787 --mutation M1A
```

### Batch query from file

```bash
python run_pathos.py --file example_input.txt --output results.csv
```

### Filter results by score

```bash
# Keep only highly pathogenic variants
python run_pathos.py --protein P51787 --min-score 0.9 --output pathogenic.csv

# Keep only highly benign variants
python run_pathos.py --protein P51787 --max-score 0.1 --output benign.csv

# Keep variants in a specific range
python run_pathos.py --protein P51787 --min-score 0.55 --max-score 0.65 --output uncertain.csv
```

### Full protein scan

```bash
# For proteins in database (instant)
python run_pathos.py --protein P51787 --output P51787_all.csv

# For proteins NOT in database (requires --scan, can take hours)
python run_pathos.py --protein Q96PN7 --scan --output Q96PN7_all.csv
```

### Input file format

Supports TXT, TSV, and CSV formats. Headers are auto-detected and skipped.

**TXT/TSV (space or tab-separated):**

```
P51787 M1A R56V    # Multiple mutations per line
Q9Y6X3 M1C         # Single mutation
P10635             # Full scan (all 19 substitutions per position)
```

**CSV (comma-separated):**

```csv
Protein,Mutation
P51787,M1A
P51787,L50R
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
| `-p, --protein` | UniProt protein ID (e.g., P51787) |
| `-m, --mutation` | Mutation in format like M1A (requires --protein) |
| `-f, --file` | Input file with protein IDs and mutations (TXT, TSV, or CSV) |
| `-o, --output` | Output CSV file (default: stdout for single mutation) |
| `--min-score` | Minimum PATHOS score threshold for filtering results (0.0-1.0) |
| `--max-score` | Maximum PATHOS score threshold for filtering results (0.0-1.0) |
| `--scan` | Enable de novo full protein scan (required for proteins not in database) |
| `--n-jobs` | Number of parallel workers for feature generation (default: 5) |
| `--batch-size` | Batch size for embedding generation (default: 100) |
| `--mmseqs-mem-limit` | Memory limit for mmseqs2 MSA generation (default: 8G) |
| `--batch-threshold` | Number of variants above which batched mode is enabled (default: 10000) |

## Full protein scan

PATHOS can predict scores for all possible mutations of a protein (19 substitutions × sequence length).

### Proteins in the database

For proteins already in the pre-computed database, simply omit the `--mutation` argument:

```bash
python run_pathos.py --protein P51787 --output P51787_all.csv
```

This instantly retrieves all pre-computed scores for that protein.

The list of 17,574 pre-computed human proteins (having less than 1,024 residues) is available in [`proteins_in_db.txt`](proteins_in_db.txt).

### De novo scan (proteins not in database)

For proteins **not** in the database, a full de novo scan is required. This can take several hours as it needs to:
- Generate MSA alignments
- Compute conservation scores
- Generate embeddings for thousands of mutations

To enable de novo scanning, add the `--scan` flag:

```bash
python run_pathos.py --protein Q96PN7 --scan --output Q96PN7_all.csv
```

**Warning:** De novo scans are computationally expensive. For a typical 500-residue protein, this means predicting ~9,500 mutations.

## Embeddings download
Soon available

## Citation

If you use PATHOS in your research, please cite:

Radjasandirane, R., Cretin, G., Diharce, J., de Brevern, A. G., & Gelly, J. C. (2026). PATHOS: Predicting Variant Pathogenicity by Combining Protein Language Models and Biological Features. Artificial Intelligence in the Life Sciences, 100165.


## Contact

For bug reports, feature requests, or questions, please contact:

radja.ragou@gmail.com
