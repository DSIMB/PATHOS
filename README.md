# PATHOS - Predicting variant pathogenicity by combining Protein Language Models and biological features

PATHOS predicts pathogenicity of protein variants using protein language models (ESM-C 600M, Ankh2 Large). Precomputed scores for 216M mutations across 20,416 human proteins.

## Installation

```bash
git clone https://github.com/DSIMB/PATHOS.git
cd PATHOS
./setup_pathos.sh
conda activate PATHOS_env
```

By default the script downloads only `pathos.db` (~10 GB), which is sufficient to run predictions for all 20,416 precomputed proteins. This covers the vast majority of use cases.

If you need de novo prediction for proteins absent from the database (very large proteins, or proteins added to UniProt after March 2025), run with the `all` argument to also download the allele frequency database, MSA alignments, and the mmseqs2 mammalian database:

```bash
./setup_pathos.sh all
```

**Disk space requirements:**

| Mode | Size |
|------|------|
| Default (`pathos.db` only) | ~10 GB |
| Full (`all`) | ~35 GB |


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
python run_pathos.py --protein I3L3L1 --scan --output I3L3L1_all.csv
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

Most queries are served instantly from the precomputed database (216M mutations, 20,416 human proteins). De novo prediction is only needed for proteins absent from the database: very large proteins that exceed the model input length, and proteins added to UniProt after March 2025 (the version used to build the database, `uniprotsp_human_20032025_can_isoforms.fasta`), such as I3L3L1.

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

For proteins already in the precomputed database, simply omit the `--mutation` argument:

```bash
python run_pathos.py --protein P51787 --output P51787_all.csv
```

This instantly retrieves all pre-computed scores for that protein.

The full list of 20,416 precomputed proteins is available in [`proteins_in_db.txt`](proteins_in_db.txt).

### De novo scan (proteins not in database)

For proteins not in the database, a full de novo scan requires generating MSA alignments, computing conservation scores, and running embeddings for every possible mutation. This can take several hours. Add the `--scan` flag to enable it:

```bash
# I3L3L1 was added to UniProt after March 2025 and is not in the precomputed database
python run_pathos.py --protein I3L3L1 --scan --output I3L3L1_all.csv
```

For a typical 500-residue protein this means computing predictions for roughly 9,500 mutations.

## PLM embeddings

Precomputed embeddings for all 216M mutations (and wild-type residues) across the 20,416 proteins are available on HuggingFace: https://huggingface.co/datasets/DSIMB/PATHOS-PLM-EMBEDDINGS

The dataset covers three protein language models:

| Model | Embedding dim | Mutation rows | Wild-type rows | Size |
|-------|--------------|---------------|----------------|------|
| ESM-C 600M | 1152 | 216.2M | 11.4M | 1.90 TiB |
| ESM-2 650M | 1280 | 216.2M | 11.4M | 2.10 TiB |
| Ankh2 Large | 1536 | 216.2M | 11.4M | 2.51 TiB |

Each row stores a position-specific embedding (`emb`) extracted at the mutated or wild-type residue, along with a mean-pooled full-sequence embedding (`mean`). Files are in Parquet format distributed across shards.

```python
from datasets import load_dataset

# Mutation embeddings (streaming recommended given the size)
ds = load_dataset("DSIMB/PATHOS-PLM-EMBEDDINGS", "esmc_600m", streaming=True)

# Wild-type embeddings
wt_ds = load_dataset("DSIMB/PATHOS-PLM-EMBEDDINGS", "wt_esmc_600m", streaming=True)
```

DuckDB can query individual shards directly without downloading the full dataset:

```python
import duckdb

df = duckdb.sql("""
    SELECT protein_id, variation, emb, mean
    FROM 'hf://datasets/DSIMB/PATHOS-PLM-EMBEDDINGS/esmc_600m/data/*.parquet'
    WHERE protein_id = 'P51787'
""").df()
```

Having precomputed embeddings for every possible human missense substitution opens several downstream uses beyond pathogenicity prediction:

- **Training and benchmarking variant effect predictors** without running PLMs at inference time. The embeddings serve as fixed input features for any supervised or self-supervised model, cutting compute costs significantly for large-scale training.
- **Studying mutational landscapes** of individual proteins or protein families. Comparing wild-type and mutant embeddings per position gives a residue-level view of how each substitution shifts the sequence representation, which correlates with structural and functional sensitivity.
- **Clustering variants by embedding similarity** to identify functionally equivalent substitutions or to stratify variants before downstream analysis.
- **Exploring the geometry of pathogenic versus benign variants** in embedding space, which can reveal model-agnostic signatures of deleteriousness shared across proteins.

Each PLM configuration inherits the license of its source model: MIT for ESM-2, CC BY-NC-SA 4.0 for Ankh2, and the Cambrian Non-Commercial license for ESM-C. Non-commercial restrictions apply when using the Ankh2 or ESM-C configs.

## Citation

If you use PATHOS or PLM embeddings in your research, please cite:

Radjasandirane, R., Cretin, G., Diharce, J., de Brevern, A. G., & Gelly, J. C. (2026). PATHOS: Predicting Variant Pathogenicity by Combining Protein Language Models and Biological Features. Artificial Intelligence in the Life Sciences, 100165.


## Contact

For bug reports, feature requests, or questions, please contact:

radja.ragou@gmail.com
