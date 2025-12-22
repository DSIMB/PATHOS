# PATHOS - Protein variant Analysis Through Human-Optimized Scoring

PATHOS is a machine learning-based tool for predicting the pathogenicity of protein variants. It uses protein language models to analyze the effects of amino acid substitutions on protein function.

## Overview

PATHOS provides pathogenicity scores for single amino acid variants (missense mutations) across the human proteome. The tool uses embeddings from state-of-the-art protein language models (ESM-C 600M, Ankh2 Large, ESM2) to predict whether a mutation is likely to be pathogenic.

### Key Features

- **Pre-computed pathogenicity scores** for millions of human protein variants
- **Command-line interface** for querying the database
- **Variant prediction pipeline** for custom mutations
- **SQLite database** for fast local queries (9.7 GB, 139M+ variants, 17,574 proteins)
- Support for single and multiple mutations per protein

## Quick Start

Get started in 2 minutes!

### Prerequisites
- Python 3.8+ (no additional dependencies required)
- Database is included: `database/pathos.db` (9.7 GB)

### Basic Commands

```bash
# Query a single protein (top 10 results)
python query_pathos.py --protein P16501 --limit 10

# Query a specific mutation
python query_pathos.py --protein Q9Y6X3 --mutation M1A

# Query from input file
python query_pathos.py --file example_input.txt

# Export to CSV
python query_pathos.py --protein P16501 --min-score 0.9 --output results.csv

# Database statistics
python query_pathos.py --stats

# List all proteins
python query_pathos.py --list-proteins

# Get help
python query_pathos.py --help
```

## Database Structure

The PATHOS database contains pre-computed predictions stored in two formats:

### 1. SQLite Database (`pathos.db`)

**Schema:**
```sql
CREATE TABLE mutations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    protein_id TEXT NOT NULL,        -- UniProt ID
    mutation TEXT NOT NULL,           -- e.g., "M1A", "R56V"
    score REAL NOT NULL,              -- PATHOS pathogenicity score (0-1)
    UNIQUE(protein_id, mutation)
);
```

**Example query:**
```bash
sqlite3 database/pathos.db "SELECT mutation, score FROM mutations WHERE protein_id='P16501' LIMIT 10;"
```

### 2. CSV Files

Individual CSV files for each protein contain detailed predictions:
- `{UniProtID}_predictions.csv`

**Columns:**
- `ID`: UniProt ID
- `Gene`: Gene name
- `Variation`: Mutation notation (e.g., "M1A")
- `esmc_600m_PATHO_pred`: ESM-C pathogenicity prediction
- `esmc_600m_ddG_pred`: Predicted stability change (ΔΔG)
- `esmc_600m_GOF_pred`: Gain-of-function prediction

## Installation

```bash
git clone https://github.com/yourusername/PATHOS.git
cd PATHOS
```

No dependencies needed! The script uses only Python standard library. Database is already included.

## Usage Examples

### Using the Query Script

#### 1. Query all mutations for a protein

```bash
python query_pathos.py --protein P16501
```

Output:
```
=== 19000 mutations for P16501 ===
Mutation  Score   Interpretation
--------------------------------
M1P       0.9876  PATHOGENIC
R56X      0.9543  PATHOGENIC
...
```

#### 2. Query a specific mutation

```bash
python query_pathos.py --protein P16501 --mutation M1A
```

Output:
```
P16501 M1A: 0.9089 (PATHOGENIC)
```

#### 3. Query from an input file

Create `variants.txt`:
```
P16501
Q9Y6X3 M1A M1C
P10635 R56V
```

Run the query:
```bash
python query_pathos.py --file variants.txt
```

#### 4. Filter by pathogenicity score

Get only highly pathogenic variants (score > 0.9):
```bash
python query_pathos.py --protein P16501 --min-score 0.9
```

Get likely benign variants (score < 0.3):
```bash
python query_pathos.py --protein P16501 --max-score 0.3
```

#### 5. Limit number of results

Get top 10 most pathogenic variants:
```bash
python query_pathos.py --protein P16501 --limit 10
```

#### 6. Export results to CSV

```bash
python query_pathos.py --protein P16501 --output results.csv
python query_pathos.py --file variants.txt --output batch_results.csv
```

#### 7. Database statistics

```bash
python query_pathos.py --stats
```

Output:
```
=== PATHOS Database Statistics ===
Total proteins:              17,574
Total mutations:             139,630,279
Avg mutations per protein:   7,945.3
Score range:                 0.0003 - 0.9992
Average score:               0.5908
```

### Advanced Examples

#### Find pathogenic variants for multiple proteins

Create `high_risk.txt`:
```
P16501
Q9Y6X3
P10635
```

Run:
```bash
python query_pathos.py --file high_risk.txt --min-score 0.9 --output high_risk_variants.csv
```

#### Query specific mutations from a file

Create `specific_variants.txt`:
```
P16501 M1A R56V W695F
Q9Y6X3 M1C A100T
P10635 L234P G456D
```

Run:
```bash
python query_pathos.py --file specific_variants.txt --output specific_results.csv
```

#### Filter variants in uncertain range

```bash
python query_pathos.py --protein P16501 --min-score 0.3 --max-score 0.7
```

### Input File Format

The input file supports two formats:

**Format 1: Protein only (returns all mutations)**
```
P16501
Q9Y6X3
P10635
```

**Format 2: Protein + specific mutations**
```
P16501 M1A R56V
Q9Y6X3 M1C
P10635 L234P G456D H789N
```

**Mixed format**
```
# Get all mutations for P16501
P16501

# Get specific mutations for Q9Y6X3
Q9Y6X3 M1A M1C

# Comments and empty lines are ignored

# Get specific mutations for P10635
P10635 R56V
```

### Direct SQLite Queries

You can also query the database directly using SQLite:

**Get all predictions for a protein:**
```bash
sqlite3 database/pathos.db \
  "SELECT mutation, score FROM mutations WHERE protein_id='P16501' ORDER BY score DESC;"
```

**Find highly pathogenic variants:**
```bash
sqlite3 database/pathos.db \
  "SELECT protein_id, mutation, score FROM mutations 
   WHERE score > 0.9 ORDER BY score DESC LIMIT 100;"
```

### Using Python

```python
import sqlite3

# Connect to database
conn = sqlite3.connect('database/pathos.db')
cursor = conn.cursor()

# Query mutations for a protein
protein_id = 'P16501'
cursor.execute("SELECT mutation, score FROM mutations WHERE protein_id=?", (protein_id,))

for mutation, score in cursor.fetchall():
    print(f"{mutation}: {score:.4f}")

conn.close()
```

## Command-Line Options

```
usage: query_pathos.py [-h] (--protein PROTEIN | --file FILE | --stats | --list-proteins)
                       [--mutation MUTATION] [--min-score MIN_SCORE] [--max-score MAX_SCORE]
                       [--output OUTPUT] [--database DATABASE] [--limit LIMIT]

Options:
  -h, --help            Show help message
  
  Input (choose one):
  --protein, -p         UniProt ID to query
  --file, -f            Input file with variants
  --stats               Show database statistics
  --list-proteins       List all proteins
  
  Query filters:
  --mutation, -m        Specific mutation (with --protein)
  --min-score           Minimum pathogenicity score
  --max-score           Maximum pathogenicity score
  --limit, -l           Limit number of results
  
  Output:
  --output, -o          Export to CSV file
  --database, -d        Database path (default: database/pathos.db)
```

## Tips & Best Practices

1. **For large queries**: Always use `--output` to save results to CSV for further analysis
2. **Finding pathogenic variants**: Use `--min-score 0.9` to focus on high-confidence predictions
3. **Batch processing**: Create input files with multiple proteins for efficient querying
4. **Score filtering**: Combine `--min-score` and `--max-score` to find variants in specific ranges
5. **Quick checks**: Use `--limit 10` to preview results before full query

## Output Interpretation

### Pathogenicity Scores

PATHOS scores range from **0 to 1**:

- **< 0.63**: BENIGN - Likely harmless
- **≥ 0.63**: PATHOGENIC - Likely disease-causing

### Additional Predictions

- **ddG (ΔΔG)**: Predicted change in protein stability (higher = more destabilizing)
- **GOF**: Gain-of-function prediction score

### CSV Output Format

When using `--output`, results are saved as CSV with columns:
- `Protein`: UniProt ID
- `Mutation`: Mutation notation
- `Score`: Pathogenicity score (0-1)
- `Interpretation`: Categorical classification

## Python API Examples

### Example 1: Find all pathogenic variants in CYP2D6 (P10635)

```bash
sqlite3 database/pathos.db << EOF
.mode column
.headers on
SELECT mutation, score 
FROM mutations 
WHERE protein_id='P10635' AND score > 0.9 
ORDER BY score DESC 
LIMIT 20;
EOF
```

### Example 2: Batch query multiple proteins

```python
import sqlite3
import pandas as pd

proteins = ['P16501', 'Q9Y6X3', 'P10635']
results = []

conn = sqlite3.connect('database/pathos.db')

for protein_id in proteins:
    df = pd.read_sql_query(
        "SELECT protein_id, mutation, score FROM mutations WHERE protein_id=?",
        conn, params=(protein_id,)
    )
    results.append(df)

all_results = pd.concat(results)
conn.close()

print(all_results.head())
```

### Example 3: Export predictions for specific mutations

```python
import sqlite3
import csv

# List of variants to query
variants = [
    ('P16501', 'M1A'),
    ('P16501', 'R56V'),
    ('Q9Y6X3', 'M1C'),
]

conn = sqlite3.connect('database/pathos.db')
cursor = conn.cursor()

with open('query_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Protein', 'Mutation', 'Score'])
    
    for protein_id, mutation in variants:
        cursor.execute(
            "SELECT score FROM mutations WHERE protein_id=? AND mutation=?",
            (protein_id, mutation)
        )
        result = cursor.fetchone()
        if result:
            writer.writerow([protein_id, mutation, result[0]])

conn.close()
print("Results exported to query_results.csv")
```

## Database Statistics

Check database coverage:
```bash
# Count total proteins in database
sqlite3 database/pathos.db "SELECT COUNT(DISTINCT protein_id) FROM mutations;"

# Count total variants
sqlite3 database/pathos.db "SELECT COUNT(*) FROM mutations;"

# List all available proteins
sqlite3 database/pathos.db "SELECT DISTINCT protein_id FROM mutations ORDER BY protein_id;"
```

## Troubleshooting

### Common Issues

**"Database not found" error:**
```bash
# Specify database location explicitly
python query_pathos.py --protein P16501 --database /path/to/pathos.db
```

**No results for a protein:**
```bash
# Check if protein is in database
python query_pathos.py --list-proteins | grep P16501
```

**Large result sets:**
```bash
# Use limit to avoid overwhelming output
python query_pathos.py --protein P16501 --limit 100

# Or export to CSV
python query_pathos.py --protein P16501 --output results.csv
```

## What's in This Repository

- `query_pathos.py` - Main command-line query script
- `database/pathos.db` - SQLite database with predictions (9.7 GB)
- `example_input.txt` - Example input file format
- `README.md` - This documentation

## Citation

If you use PATHOS in your research, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]

## Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: [Your contact information]

## Related Resources

- [UniProt](https://www.uniprot.org/) - Protein sequence and annotation database
- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) - Database of genomic variation and human health
- [ESM](https://github.com/facebookresearch/esm) - Evolutionary Scale Modeling

---

**Note:** This is a command-line interface for querying the PATHOS database. For the web server version, see the main PATHOS repository.
