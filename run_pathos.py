#!/usr/bin/env python3
"""
PATHOS Prediction Runner

This script runs PATHOS predictions for variants not already in the database.
It validates input mutations, generates embeddings, runs inference, and combines
results from both database queries and de novo predictions.

Usage:
    python run_pathos.py --input variants.txt --output results.csv
    
Input format:
    P16501 M1A R56V    # Specific mutations
    Q9Y6X3 M1C         # Single mutation
"""

import argparse
import sqlite3
import sys
import os
import torch
import numpy as np
import pandas as pd
import requests
import time
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import T5EncoderModel, AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
# import warnings
# warnings.filterwarnings('ignore')

from ete3 import Tree
from pastml.acr import pastml_pipeline

# Get the directory where this script is located (works for git repo)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed paths relative to script location
DB_PATH = os.path.join(SCRIPT_DIR, "database", "pathos.db")
AF_SQLITE_PATH = os.path.join(SCRIPT_DIR, "database", "af_index.sqlite")
FASTA_PATH = os.path.join(SCRIPT_DIR, "database", "uniprotsp_human_20032025_can_isoforms.fasta")
TREE_PATH = os.path.join(SCRIPT_DIR, "database", "clean_mammalia.tre")
MSA_FOLDER = os.path.join(SCRIPT_DIR, "database", "MSAs")
PASTML_CACHE = os.path.join(SCRIPT_DIR, "database", "pastml_cache")
FASTA_FOLDER = os.path.join(SCRIPT_DIR, "database", "fastas")
MAMMALS_DB = os.path.join(SCRIPT_DIR, "database", "mmseqs_db", "mammalsDB")
MODELS_FOLDER = os.path.join(SCRIPT_DIR, "models")

# Trained model checkpoint files
TRAINED_MODELS = {
    "ankh2_large": "PATHOS_ankh2.ckpt",
    "esmc_600m": "PATHOS_ESMC.ckpt"
}


# Model configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLM_MODELS = ["esmc_600m", "ankh2_large"]
PLM_EMBEDDING_DIMS = {"esmc_600m": 1152, "ankh2_large": 1536}

# Feature transformation parameters (from training data)
PARAM_PASTML = {'log_min': -9.400765740838802, 'log_max': 1.4426951595367387e-09}                                                                                                                                
PARAM_AF = {'log_min': -20.477300046830425, 'log_max': -1.460410576505362e-06}
PARAM_STRING = {'log_min': -10.629354334852547, 'log_max': -0.8828422962808098}

# Amino acid alphabet
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


class FC_model(nn.Module):
    """PATHOS prediction model"""
    def __init__(self, input_size=6):
        super(FC_model, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))

# ============================================================================
# FEATURE GENERATION FUNCTIONS
# ============================================================================

def transform_log2_minmax(data, params):
    """Apply log2-minmax transformation to features"""
    epsilon = 1e-9
    data_log = np.log2(data + epsilon)
    return (data_log - params['log_min']) / (params['log_max'] - params['log_min'])


# ============================================================================
# PASTML CONSERVATION SCORE COMPUTATION
# ============================================================================

def check_and_generate_msas(protein_ids: List[str], msa_folder: str, fasta_folder: str, 
                            mammals_db: str, mem_limit: str = "8G", debug: bool = False) -> Tuple[List[str], List[str], List[str]]:
    """Check MSA availability for all proteins and generate missing ones
    
    Args:
        protein_ids: List of UniProt IDs to check
        msa_folder: Path to MSA folder
        fasta_folder: Path to individual FASTA files
        mammals_db: Path to mammalsDB for mmseqs2
        mem_limit: Memory limit for mmseqs2 (default: 8G)
        debug: Print debug information (default: False)
    
    Returns:
        Tuple of (available_msas, generated_msas, failed_msas)
    """
    import shutil
    
    available = []
    to_generate = []
    
    # Check which MSAs exist
    for protein_id in protein_ids:
        msa_file = os.path.join(msa_folder, protein_id)
        if os.path.exists(msa_file):
            available.append(protein_id)
        else:
            to_generate.append(protein_id)
    
    # Print summary before generation
    print(f"    MSA already available: {len(available)}/{len(protein_ids)}")
    print(f"    MSA to generate: {len(to_generate)}/{len(protein_ids)}")
    
    generated = []
    failed = []
    
    if to_generate:
        # Check if mmseqs is available
        if shutil.which('mmseqs') is None:
            print(f"    WARNING: mmseqs not found, cannot generate MSAs")
            failed = to_generate
        elif not os.path.exists(mammals_db):
            print(f"    WARNING: mammalsDB not found at {mammals_db}, cannot generate MSAs")
            failed = to_generate
        else:
            # Generate missing MSAs
            print(f"    Generating {len(to_generate)} MSAs with mmseqs2 (mem_limit={mem_limit})...")
            for protein_id in tqdm(to_generate, desc="    Generating MSAs", dynamic_ncols=True):
                result = generate_msa_with_mmseqs(protein_id, fasta_folder, msa_folder, mammals_db, 
                                                   mem_limit=mem_limit, debug=debug)
                if result:
                    generated.append(protein_id)
                else:
                    failed.append(protein_id)
                    if debug:
                        print(f"      Failed to generate MSA for {protein_id}")
    
    return available, generated, failed


def generate_msa_with_mmseqs(protein_id: str, fasta_folder: str, msa_folder: str, 
                             mammals_db: str, mem_limit: str = "8G", debug: bool = False) -> Optional[str]:
    """Generate MSA using mmseqs2 if not already available
    
    Args:
        protein_id: UniProt ID
        fasta_folder: Path to folder containing individual FASTA files
        msa_folder: Path to MSA output folder
        mammals_db: Path to mammalsDB (without extension)
        mem_limit: Memory limit for mmseqs2 (default: 8G)
        debug: Print debug information (default: False)
    
    Returns:
        Path to generated MSA file, or None if generation failed
    """
    import subprocess
    import shutil
    
    def debug_print(msg):
        if debug:
            print(f"      [DEBUG mmseqs] {msg}")
    
    # Check if mmseqs is available
    if shutil.which('mmseqs') is None:
        debug_print("mmseqs not found in PATH")
        return None
    
    fasta_file = os.path.join(fasta_folder, f"{protein_id}.fasta")
    if not os.path.exists(fasta_file):
        debug_print(f"FASTA file not found: {fasta_file}")
        return None
    
    if not os.path.exists(mammals_db):
        debug_print(f"mammalsDB not found: {mammals_db}")
        return None
    
    # Output paths - use a temporary folder for intermediate files
    temp_folder = os.path.join(msa_folder, f".tmp_{protein_id}")
    os.makedirs(temp_folder, exist_ok=True)
    
    path_queryDB = os.path.join(temp_folder, "queryDB")
    result_prefix = os.path.join(temp_folder, f"result_{protein_id}")
    tmp_folder = os.path.join(temp_folder, "tmp")
    msa_result = os.path.join(temp_folder, f"result_{protein_id}_msa")
    unpack_dir = os.path.join(temp_folder, "unpack")
    
    # Final MSA path directly in msa_folder
    final_msa = os.path.join(msa_folder, protein_id)
    
    debug_print(f"Starting MSA generation for {protein_id}")
    debug_print(f"  FASTA: {fasta_file}")
    debug_print(f"  Temp folder: {temp_folder}")
    debug_print(f"  Final MSA: {final_msa}")
    debug_print(f"  Memory limit: {mem_limit}")
    
    try:
        # Step 1: Create query database
        debug_print("Step 1/5: Creating query database...")
        result = subprocess.run(
            ['mmseqs', 'createdb', fasta_file, path_queryDB],
            check=True, capture_output=True, text=True
        )
        if debug and result.stderr:
            debug_print(f"  stderr: {result.stderr[:200]}")
        
        # Step 2: Search against mammalsDB (with memory limit)
        debug_print("Step 2/5: Searching against mammalsDB...")
        result = subprocess.run(
            ['mmseqs', 'search', path_queryDB, mammals_db, result_prefix, tmp_folder,
             '--max-seqs', '5000', '--min-seq-id', '0.5', '--split-memory-limit', mem_limit],
            check=True, capture_output=True, text=True
        )
        if debug and result.stderr:
            debug_print(f"  stderr: {result.stderr[:200]}")
        
        # Step 3: Convert alignments
        debug_print("Step 3/5: Converting alignments...")
        result = subprocess.run(
            ['mmseqs', 'convertalis', path_queryDB, mammals_db, result_prefix, f"{result_prefix}.m8"],
            check=True, capture_output=True, text=True
        )
        if debug and result.stderr:
            debug_print(f"  stderr: {result.stderr[:200]}")
        
        # Step 4: Generate MSA
        debug_print("Step 4/5: Generating MSA...")
        result = subprocess.run(
            ['mmseqs', 'result2msa', path_queryDB, mammals_db, result_prefix, msa_result],
            check=True, capture_output=True, text=True
        )
        if debug and result.stderr:
            debug_print(f"  stderr: {result.stderr[:200]}")
        
        # Step 5: Unpack MSA
        debug_print("Step 5/5: Unpacking MSA...")
        result = subprocess.run(
            ['mmseqs', 'unpackdb', msa_result, unpack_dir, '--unpack-name-mode', '0'],
            check=True, capture_output=True, text=True
        )
        if debug and result.stderr:
            debug_print(f"  stderr: {result.stderr[:200]}")
        
        # Step 6: Move final MSA to msa_folder and cleanup
        msa_output = os.path.join(unpack_dir, "0")
        
        if os.path.exists(msa_output):
            shutil.move(msa_output, final_msa)
            # Remove temporary folder
            shutil.rmtree(temp_folder, ignore_errors=True)
            debug_print(f"MSA generated successfully: {final_msa}")
            return final_msa
        
        debug_print(f"MSA output file not found: {msa_output}")
        shutil.rmtree(temp_folder, ignore_errors=True)
        return None
        
    except subprocess.CalledProcessError as e:
        debug_print(f"mmseqs command failed: {e.cmd}")
        debug_print(f"  Return code: {e.returncode}")
        debug_print(f"  stderr: {e.stderr[:500] if e.stderr else 'None'}")
        shutil.rmtree(temp_folder, ignore_errors=True)
        return None
    except Exception as e:
        debug_print(f"Exception: {type(e).__name__}: {e}")
        shutil.rmtree(temp_folder, ignore_errors=True)
        return None


def get_msa_fasta(msa_file: str) -> Dict[str, str]:
    """Load MSA sequences from FASTA file
    
    Expects FASTA headers with organism names like:
    >Homo_sapiens OS=...
    """
    dict_seq = {}
    get = False
    
    if not os.path.exists(msa_file):
        return {}
    
    with open(msa_file) as fin:
        for line in fin:
            line = line.strip()
            if line.startswith(">"):
                if "OS" not in line:
                    get = False
                    continue
                try:
                    org = "_".join(line.split("=")[1].split(" OX")[0].split())
                    if org in dict_seq:
                        get = False
                        continue
                    dict_seq[org] = ""
                    get = True
                except:
                    get = False
                    continue
            elif get:
                dict_seq[org] += line
    
    return dict_seq


def pos_to_msa_index(sequence: str, position: int) -> int:
    """Convert protein position to MSA index (accounting for gaps)"""
    index = 0
    for i, aa in enumerate(sequence):
        if aa in AA_ALPHABET:
            index += 1
        if index == position:
            return i
    return -1


def create_pastml_annotation(dict_fasta: Dict[str, str], protein_id: str, 
                             mutation: str, annotation_dir: str):
    """Create annotation CSV file for PASTML"""
    wt_aa, position, mut_aa = parse_mutation(mutation)
    
    if "Homo_sapiens" not in dict_fasta:
        return False
    
    msa_index = pos_to_msa_index(dict_fasta["Homo_sapiens"], position)
    if msa_index == -1:
        return False
    
    os.makedirs(annotation_dir, exist_ok=True)
    annot_file = os.path.join(annotation_dir, f"{protein_id}_{mutation}_annot.csv")
    
    with open(annot_file, "w") as f:
        f.write("organism,residue\n")
        for org in dict_fasta:
            if org == "Homo_sapiens":
                # Use the mutant amino acid for human
                f.write(f"{org},{mut_aa}\n")
            else:
                # Use the MSA amino acid for other organisms
                aa = dict_fasta[org][msa_index] if msa_index < len(dict_fasta[org]) else '-'
                f.write(f"{org},{aa}\n")
    
    return True


def prune_phylo_tree(tree: 'Tree', msa_organisms: List[str], output_file: str) -> 'Tree':
    """Prune phylogenetic tree to match organisms in MSA"""
    # Get tree node names from all nodes (not just leaves)
    all_nodes = tree.get_descendants() + [tree]
    tree_node_names = {node.name for node in all_nodes}
    
    # Keep only organisms present in both tree and MSA
    keep_org = [org for org in msa_organisms if org in tree_node_names]
    
    if len(keep_org) < 2:
        return None
    
    pruned_tree = tree.copy()
    pruned_tree.prune(keep_org)  # Don't preserve branch length
    
    # Write pruned tree
    with open(output_file, "w") as f:
        f.write(pruned_tree.write(format=1))
    
    return pruned_tree


def run_pastml_inference(tree_file: str, annot_file: str, output_dir: str) -> bool:
    """Run PASTML ancestral sequence reconstruction"""
    try:
        pastml_pipeline(
            data=annot_file,
            data_sep=',',
            columns=['residue'],
            name_column='residue',
            tree=tree_file,
            work_dir=output_dir,
            model="JC",
            verbose=False
        )
        return True
    except Exception as e:
        return False


def compute_pastml_score(protein_id: str, mutation: str, msa_folder: str,
                        tree: 'Tree', tree_nodes: Set[str], cache_dir: str,
                        fasta_folder: str = None, mammals_db: str = None) -> float:
    """Compute PASTML conservation score for a single variant
    
    Returns probability of mutation at ancestral node (0-1).
    Higher values = mutation is common in evolution = less pathogenic.
    Low values (close to 0) = mutation is rare = likely pathogenic.
    
    If MSA is not available and fasta_folder/mammals_db are provided,
    will attempt to generate MSA using mmseqs2.
    """
    wt_aa, position, mut_aa = parse_mutation(mutation)
    
    # Check cache first
    cache_file = os.path.join(cache_dir, protein_id, f"{mutation}_pastml.txt")
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached_value = float(f.read().strip())
                return cached_value
        except:
            pass
    
    # Setup directories
    annotation_dir = os.path.join(cache_dir, protein_id, mutation)
    os.makedirs(annotation_dir, exist_ok=True)
    
    # Check if already computed
    prob_file = os.path.join(annotation_dir, "marginal_probabilities.character_residue.model_JC.tab")
    
    if not os.path.exists(prob_file):
        # Load MSA
        msa_file = os.path.join(msa_folder, protein_id)
        
        if not os.path.exists(msa_file):
            # Try to generate MSA with mmseqs2
            if fasta_folder and mammals_db:
                msa_file = generate_msa_with_mmseqs(protein_id, fasta_folder, msa_folder, mammals_db)
                if not msa_file:
                    return np.nan  # MSA generation failed
            else:
                return np.nan  # No MSA available
        
        dict_fasta = get_msa_fasta(msa_file)
        if not dict_fasta or "Homo_sapiens" not in dict_fasta:
            return np.nan
        
        # Create annotation file
        if not create_pastml_annotation(dict_fasta, protein_id, mutation, annotation_dir):
            return np.nan
        
        # Prune tree
        pruned_tree_file = os.path.join(annotation_dir, f"{protein_id}_pruned.tre")
        msa_organisms = list(dict_fasta.keys())
        pruned_tree = prune_phylo_tree(tree, msa_organisms, pruned_tree_file)
        
        if pruned_tree is None:
            return np.nan
        
        # Run PASTML
        annot_file = os.path.join(annotation_dir, f"{protein_id}_{mutation}_annot.csv")
        if not run_pastml_inference(pruned_tree_file, annot_file, annotation_dir):
            return np.nan
    
    # Read probability from output
    try:
        df_pastml = pd.read_csv(prob_file, sep="\t")
        
        # Get ancestral node for Homo sapiens
        tree_file = os.path.join(annotation_dir, f"named.tree_{protein_id}_pruned.nwk")
        result_tree = Tree(tree_file, format=1)
        
        human_node = result_tree.search_nodes(name="Homo_sapiens")[0]
        parent_node_name = human_node.up.name
        
        # Extract probability for the mutant amino acid
        prob = df_pastml[df_pastml["node"] == parent_node_name][mut_aa].values[0]
        
        # Cache result
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w") as f:
            f.write(str(prob))
        
        return float(prob)
        
    except Exception as e:
        return np.nan


# ============================================================================
# UNIPROT ANNOTATIONS (GFF-based implementation)
# ============================================================================


def download_gff_from_uniprot(protein_id: str, output_path: str) -> bool:
    """Download GFF annotation file from UniProt API
    
    Args:
        protein_id: UniProt accession ID
        output_path: Where to save the GFF file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.gff"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(response.text)
            return True
        else:
            return False
    except Exception as e:
        return False


def get_matrix_annot(protein_id: str, data_dir: str) -> Optional[np.ndarray]:
    """Create annotation matrix from GFF file for a protein
    
    This implements the exact logic from generate_inputs.py:
    - Creates a matrix of shape (max_seq_len, 18) for 18 selected annotation types
    - Parses GFF file and marks positions with annotations
    - Special handling for 'Disulfide bond' (marks both start and end positions)
    - Trims matrix to last non-zero row
    
    Args:
        protein_id: UniProt accession ID
        data_dir: Base data directory
    
    Returns:
        Annotation matrix of shape (seq_length, 18) or None if GFF not found
    """
    # Define the 18 selected annotation types used by PATHOS (in order)
    dict_index = {
        'Beta strand': 0,
        'Helix': 1,
        'Natural variant': 2,
        'Topological domain': 3,
        'Mutagenesis': 4,
        'Domain': 5,
        'Region': 6,
        'Alternative sequence': 7,
        'Turn': 8,
        'DNA binding': 9,
        'Site': 10,
        'Sequence conflict': 11,
        'Disulfide bond': 12,
        'Repeat': 13,
        'Binding site': 14,
        'Transmembrane': 15,
        'Intramembrane': 16,
        'Modified residue': 17
    }
    
    # Try to find GFF file in multiple locations (same path as generate_inputs.py)
    gff_paths = [
        os.path.join(data_dir, "uniprot", f"{protein_id}.gff"),
        os.path.join(data_dir, "gff", f"{protein_id}.gff"),
        os.path.join(data_dir, f"{protein_id}.gff")
    ]
    
    gff_file = None
    for path in gff_paths:
        if os.path.exists(path):
            gff_file = path
            break
    
    # If not found, try to download it
    if not gff_file:
        download_path = os.path.join(data_dir, "uniprot", f"{protein_id}.gff")
        if download_gff_from_uniprot(protein_id, download_path):
            gff_file = download_path
        else:
            return None
    
    # Parse GFF file
    try:
        annot_matrix = np.zeros((40000, 18), dtype=int)
        
        with open(gff_file) as f:
            f.readline()  # Skip header
            for line in f:
                if line.startswith('#'):
                    continue
                
                items = line.strip().split("\t")
                if len(items) < 5:
                    continue
                
                # Check if "isoform" is in the description (last columns)
                # GFF format has 9 columns, with attributes in column 9 (index 8)
                if len(items) >= 9:
                    # Check all columns from position 8 onwards for "isoform"
                    description = "\t".join(items[8:]).lower()
                    if "isoform" in description:
                        continue  # Skip this annotation
                
                annot = items[2]
                if annot not in dict_index:
                    continue
                
                try:
                    start_pos = int(items[3])
                    end_pos = int(items[4])
                    annot_index = dict_index[annot]
                    
                    if annot != "Disulfide bond":
                        # Mark all positions in the range
                        annot_matrix[start_pos - 1:end_pos, annot_index] = 1
                    else:
                        # For disulfide bonds, mark only the two endpoints
                        annot_matrix[start_pos - 1, annot_index] = 1
                        annot_matrix[end_pos - 1, annot_index] = 1
                except (ValueError, IndexError) as e:
                    continue
        
        # Trim matrix to the last non-zero row
        nonzero_rows = np.where(np.any(annot_matrix != 0, axis=1))[0]
        if len(nonzero_rows) == 0:
            return np.zeros((1, 18), dtype=int)
        
        last_nonzero_row = np.max(nonzero_rows)
        trimmed_matrix = annot_matrix[:last_nonzero_row + 1]
        
        return trimmed_matrix
        
    except Exception as e:
        return None


def window_annot(mutation: str, matrix: np.ndarray, window_size: int = 5) -> List[int]:
    """Extract annotation window around mutation position
    
    This implements the exact logic from generate_inputs.py:
    - Extracts a window of ±window_size residues around the mutation
    - Adjusts window if near sequence start/end
    - Sums annotations across all positions in the window
    - Returns binary vector (!=0)
    
    Args:
        mutation: Mutation string (e.g., 'R50K')
        matrix: Annotation matrix from get_matrix_annot (seq_length, 18)
        window_size: Half-window size (default 5 = ±5 residues = 11 total)
    
    Returns:
        List of 18 binary values (1 if annotation present in window, 0 otherwise)
    """
    # Extract position from mutation (1-based)
    wt_aa, position, mut_aa = parse_mutation(mutation)
    position_idx = position - 1  # Convert to 0-based index
    
    total_length = matrix.shape[0]
    
    left_part = window_size
    right_part = window_size
    
    # Adjust window if near the start
    if position_idx - left_part < 0:
        excess = left_part - position_idx
        left_part = position_idx
        right_part += excess
    
    # Adjust window if near the end
    if position_idx + right_part >= total_length:
        excess = (position_idx + right_part + 1) - total_length
        right_part -= excess
        left_part += excess
    
    # Extract window and sum across positions
    window_start = position_idx - left_part
    window_end = position_idx + right_part + 1
    
    # Sum annotations across the window
    window_sum = matrix[window_start:window_end].sum(axis=0)
    
    # Convert to binary (presence/absence)
    return (window_sum != 0).astype(int).tolist()


def load_uniprot_annotations(protein_id: str, mutation: str, sequences: Dict[str, str], 
                            data_dir: str, window_size: int = 5) -> List[int]:
    """Load UniProt annotation features for a specific variant
    
    This is the main function that combines get_matrix_annot and window_annot
    to produce the final 18-dimensional binary annotation vector.
    
    Args:
        protein_id: UniProt accession ID
        mutation: Mutation string (e.g., 'R50K')
        sequences: Dictionary of protein sequences
        data_dir: Base data directory
        window_size: Window size for annotation extraction (default 5)
    
    Returns:
        List of 18 binary values representing annotations in the window
    """
    # Get the full annotation matrix for this protein
    matrix = get_matrix_annot(protein_id, data_dir)
    
    if matrix is None:
        return [0] * 18
    
    # Extract window-based features
    annot_vector = window_annot(mutation, matrix, window_size=window_size)
    
    return annot_vector


def get_gene_name_from_uniprot(uniprot_id: str) -> Optional[str]:
    """Query UniProt API to get gene name for a UniProt ID
    
    Args:
        uniprot_id: UniProt accession ID
    
    Returns:
        Gene name or None if not found
    """
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Try to get the primary gene name
            if 'genes' in data and len(data['genes']) > 0:
                if 'geneName' in data['genes'][0]:
                    gene_name = data['genes'][0]['geneName']['value']
                    return gene_name
    except Exception as e:
        pass
    
    return None


def load_gene_names(protein_ids: List[str], data_dir: str) -> Dict[str, str]:
    """Load gene names for proteins from UniProt mapping file
    
    Args:
        protein_ids: List of UniProt IDs
        data_dir: Base data directory
    
    Returns:
        Dictionary mapping UniProt ID to gene name
    """
    # Try multiple possible locations
    mapping_paths = [
        os.path.join(data_dir, "idmapping_uniprot_gene.tsv"),
        "/lustre/fswork/projects/rech/avo/uky46my/PATHOS/idmapping_uniprot_gene.tsv",
        "/dsimb/wasabi/radjasan/these/idmapping_uniprot_gene.tsv"
    ]
    
    gene_dict = {}
    
    for mapping_file in mapping_paths:
        if os.path.exists(mapping_file):
            try:
                df = pd.read_csv(mapping_file, sep='\t', names=["UniProtID", "Entry", "Gene"], header=None)
                gene_dict = dict(zip(df["UniProtID"], df["Gene"]))
                break
            except Exception as e:
                continue
    
    return gene_dict


def load_allele_frequency(protein_id: str, mutation: str, gene_name: str, af_sqlite: str) -> float:
    """Load Allele Frequency for a specific variant from SQLite database
    
    Args:
        protein_id: UniProt ID
        mutation: Mutation string (e.g., M1A)
        gene_name: Gene name for the protein
        af_sqlite: Path to AF SQLite database
    
    Returns:
        Allele frequency or 0 if not found (NaN values are treated as 0)
    """
    if not af_sqlite or not os.path.exists(af_sqlite):
        return np.nan
    
    try:
        conn = sqlite3.connect(af_sqlite)
        cursor = conn.cursor()
        
        query = "SELECT AF FROM af_table WHERE Gene=? AND Variation=?"
        result = cursor.execute(query, (gene_name, mutation)).fetchone()
        conn.close()
        
        if result and result[0] is not None:
            return float(result[0])
    except Exception as e:
        pass
    
    return np.nan


def load_allele_frequencies_batch(variants: List[Tuple[str, str]], gene_names: Dict[str, str], af_sqlite: str) -> Dict[Tuple[str, str], float]:
    """Load Allele Frequencies for all variants in a single database query
    
    Args:
        variants: List of (protein_id, mutation) tuples
        gene_names: Dictionary mapping protein IDs to gene names
        af_sqlite: Path to AF SQLite database
    
    Returns:
        Dictionary mapping (protein_id, mutation) to allele frequency
    """
    af_scores = {}
    
    if not af_sqlite or not os.path.exists(af_sqlite):
        return {v: np.nan for v in variants}
    
    try:
        conn = sqlite3.connect(af_sqlite)
        cursor = conn.cursor()
        
        # Build list of (gene, mutation) pairs to query
        query_pairs = []
        variant_to_gene = {}
        for protein_id, mutation in variants:
            gene_name = gene_names.get(protein_id)
            if gene_name:
                query_pairs.append((gene_name, mutation))
                variant_to_gene[(protein_id, mutation)] = gene_name
        
        # Query in batches to avoid SQLite limits
        batch_size = 500
        results_dict = {}
        
        for i in range(0, len(query_pairs), batch_size):
            batch = query_pairs[i:i + batch_size]
            placeholders = ','.join(['(?, ?)'] * len(batch))
            flat_params = [item for pair in batch for item in pair]
            
            query = f"SELECT Gene, Variation, AF FROM af_table WHERE (Gene, Variation) IN (VALUES {placeholders})"
            cursor.execute(query, flat_params)
            
            for row in cursor.fetchall():
                gene, variation, af = row
                results_dict[(gene, variation)] = float(af) if af is not None else np.nan
        conn.close()
        
        # Map back to (protein_id, mutation) keys
        for protein_id, mutation in variants:
            gene_name = variant_to_gene.get((protein_id, mutation))
            if gene_name and (gene_name, mutation) in results_dict:
                af_scores[(protein_id, mutation)] = results_dict[(gene_name, mutation)]
            else:
                af_scores[(protein_id, mutation)] = np.nan
                
    except Exception as e:
        # Fallback: return NaN for all
        return {v: np.nan for v in variants}
    
    return af_scores


def load_string_scores_batch(protein_ids: List[str], data_dir: str) -> Dict[str, float]:
    """Load STRING interaction scores for all proteins at once
    
    Args:
        protein_ids: List of UniProt IDs
        data_dir: Base data directory
    
    Returns:
        Dictionary mapping protein ID to STRING score
    """
    string_scores = {pid: np.nan for pid in protein_ids}
    
    # Try multiple STRING data locations
    string_paths = [
        "/dsimb/wasabi/radjasan/these/STRING/prot_prop.tsv",
        os.path.join(data_dir, "STRING", "prot_prop.tsv"),
        os.path.join(data_dir, "string_scores.tsv")
    ]
    string_path = os.path.join(data_dir, "STRING", "STIRNG_prot.tsv")
     # Load entire file once
    df = pd.read_csv(string_path, sep='\t', names=["ID", "STRING"], header=None)
    # Create lookup dict from dataframe
    string_dict = dict(zip(df['ID'], df['STRING']))
    
    # Update scores for requested proteins
    for pid in protein_ids:
        if pid in string_dict:
            string_scores[pid] = float(string_dict[pid])
                
    
    return string_scores

def _process_variant_worker(args):
    """Worker function for parallel variant processing (module-level for pickling)
    
    Args:
        args: Tuple of (protein_id, mutation, seq, af_score, string_score, 
                       do_pastml, msa_dir, tree_path, cache, data_dir, fasta_folder, mammals_db)
    
    Returns:
        Dictionary with variant features or None if processing failed
    """
    (protein_id, mutation, seq, af_score, string_score, 
     do_pastml, msa_dir, tree_path, cache, data_dir, fasta_folder, mammals_db) = args
    
    try:
        wt_aa, position, mut_aa = parse_mutation(mutation)
        
        # Get annotation features
        annot_features = load_uniprot_annotations(protein_id, mutation, {protein_id: seq}, data_dir, window_size=5)
        
        # Compute PASTML score
        if do_pastml and tree_path:
            try:
                local_tree = Tree(tree_path, format=1, quoted_node_names=False)
                local_nodes = {n.name for n in local_tree.get_descendants() + [local_tree]}
                pastml_score = compute_pastml_score(
                    protein_id, mutation, msa_dir, local_tree, local_nodes, cache,
                    fasta_folder=fasta_folder, mammals_db=mammals_db
                )
            except:
                pastml_score = np.nan
        else:
            pastml_score = np.nan
        
        return {
            'ID': protein_id,
            'Variation': mutation,
            'PASTML': pastml_score,
            'AF': af_score,
            'STRING': string_score,
            'ANNOTATIONS': annot_features
        }
    except Exception as e:
        return None


def generate_features_for_variants(
    variants: List[Tuple[str, str]],
    sequences: Dict[str, str],
    data_dir: str,
    msa_folder: str = None,
    tree_path: str = None,
    pastml_cache: str = None,
    compute_pastml: bool = False,
    af_sqlite: str = None,
    n_jobs: int = None,
    fasta_folder: str = None,
    mammals_db: str = None
) -> pd.DataFrame:
    """Generate all required features for PATHOS prediction
    
    Args:
        variants: List of (protein_id, mutation) tuples
        sequences: Dictionary mapping protein IDs to sequences
        data_dir: Base data directory
        msa_folder: Path to MSA folder for PASTML
        tree_path: Path to phylogenetic tree file
        pastml_cache: Path to PASTML cache directory
        compute_pastml: Whether to compute PASTML scores
        af_sqlite: Path to allele frequency SQLite database
        n_jobs: Number of parallel workers (default: CPU count - 1)
        fasta_folder: Path to individual FASTA files (for MSA generation)
        mammals_db: Path to mammalsDB for mmseqs2 (for MSA generation)
    """
    # Load gene names for all proteins
    unique_proteins = list(set([pid for pid, _ in variants]))
    gene_names = load_gene_names(unique_proteins, data_dir)
    
    # Pre-fetch missing gene names from UniProt API (sequential, with rate limiting)
    missing_genes = [pid for pid in unique_proteins if pid not in gene_names]
    if missing_genes:
        for pid in tqdm(missing_genes[:50], desc="Fetching gene names", dynamic_ncols=True):  # Limit API calls
            gene_name = get_gene_name_from_uniprot(pid)
            if gene_name:
                gene_names[pid] = gene_name
            time.sleep(0.2)
    
    # Load phylogenetic tree if computing PASTML
    tree = None
    tree_nodes = set()
    if compute_pastml:
        if tree_path and os.path.exists(tree_path):
            try:
                tree = Tree(tree_path, format=1, quoted_node_names=False)
                all_nodes = tree.get_descendants() + [tree]
                tree_nodes = {node.name for node in all_nodes}
            except Exception as e:
                compute_pastml = False
        else:
            compute_pastml = False
    
    # Pre-load STRING scores for all proteins in batch (single file read)
    string_scores = load_string_scores_batch(unique_proteins, data_dir)
    
    # Pre-load all AF scores in batch (single database connection)
    af_sqlite = af_sqlite or AF_SQLITE_PATH
    af_scores = load_allele_frequencies_batch(variants, gene_names, af_sqlite)
    
    # Determine number of workers
    if n_jobs is None:
        n_jobs = 5
    
    # Prepare arguments for parallel processing
    msa_folder = msa_folder or MSA_FOLDER
    pastml_cache = pastml_cache or PASTML_CACHE
    fasta_folder = fasta_folder or FASTA_FOLDER
    mammals_db = mammals_db or MAMMALS_DB
    
    rows = []
    
    # Use parallel processing if n_jobs > 1 and we have enough variants
    if n_jobs > 1 and len(variants) > 10:
        # Prepare all arguments
        all_args = []
        for protein_id, mutation in variants:
            seq = sequences.get(protein_id)
            if not seq:
                continue
            string_score = string_scores.get(protein_id, np.nan)
            af_score = af_scores.get((protein_id, mutation), np.nan)
            all_args.append((
                protein_id, mutation, seq, af_score, string_score,
                compute_pastml, msa_folder, tree_path, pastml_cache, data_dir,
                fasta_folder, mammals_db
            ))
        
        # Process in parallel using module-level function
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_process_variant_worker, args): i for i, args in enumerate(all_args)}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Features", dynamic_ncols=True):
                result = future.result()
                if result is not None:
                    rows.append(result)
    else:
        # Sequential processing for small datasets or n_jobs=1
        for protein_id, mutation in tqdm(variants, desc="Features", dynamic_ncols=True):
            sequence = sequences.get(protein_id)
            if not sequence:
                continue
            
            try:
                wt_aa, position, mut_aa = parse_mutation(mutation)
                
                # Get annotation features
                annot_features = load_uniprot_annotations(protein_id, mutation, sequences, data_dir, window_size=5)
                
                # Compute PASTML score
                if compute_pastml:
                    pastml_score = compute_pastml_score(
                        protein_id, mutation, msa_folder,
                        tree, tree_nodes, pastml_cache,
                        fasta_folder=fasta_folder, mammals_db=mammals_db
                    )
                else:
                    pastml_score = np.nan
                
                # Get pre-loaded scores
                af_score = af_scores.get((protein_id, mutation), np.nan)
                string_score = string_scores.get(protein_id, np.nan)
                
                rows.append({
                    'ID': protein_id,
                    'Variation': mutation,
                    'PASTML': pastml_score,
                    'AF': af_score,
                    'STRING': string_score,
                    'ANNOTATIONS': annot_features
                })
                
            except Exception as e:
                continue
    
    df = pd.DataFrame(rows)
    
    # Fill missing values and apply transformations
    df['STRING'] = df['STRING'].fillna(0.05996573865027195)
    df['PASTML'] = transform_log2_minmax(df['PASTML'], PARAM_PASTML)
    df['AF'] = transform_log2_minmax(df['AF'], PARAM_AF)
    df['STRING'] = transform_log2_minmax(df['STRING'], PARAM_STRING)
    df['AF'] = df['AF'].fillna(0)
    
    return df


# ============================================================================
# FASTA AND SEQUENCE UTILITIES
# ============================================================================


def load_single_fasta(fasta_file: str) -> str:
    """Load a single protein sequence from a FASTA file
    
    Args:
        fasta_file: Path to individual FASTA file
    
    Returns:
        Protein sequence as string, or empty string if not found
    """
    if not os.path.exists(fasta_file):
        return ""
    
    try:
        sequence_lines = []
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    continue  # Skip header
                sequence_lines.append(line)
        return ''.join(sequence_lines)
    except Exception as e:
        return ""


def load_fasta_sequences(fasta_path: str, protein_ids: list = None) -> Dict[str, str]:
    """Load UniProt sequences - tries individual FASTA files first, then big FASTA"""
    sequences = {}
    data_dir = os.path.dirname(fasta_path)
    fastas_dir = os.path.join(data_dir, "fastas")
    
    # Try to load individual FASTA files first
    if protein_ids:
        for protein_id in protein_ids:
            individual_fasta = os.path.join(fastas_dir, f"{protein_id}.fasta")
            sequence = load_single_fasta(individual_fasta)
            if sequence:
                sequences[protein_id] = sequence
        
        missing_proteins = [pid for pid in protein_ids if pid not in sequences]
        if not missing_proteins:
            return sequences
        protein_ids = missing_proteins
    
    # Load from main UniProtKB FASTA file
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    if protein_ids is None or current_id in protein_ids:
                        sequences[current_id] = ''.join(current_seq)
                parts = line.split('|')
                if len(parts) >= 2:
                    current_id = parts[1]
                current_seq = []
            else:
                if protein_ids is None or (current_id and current_id in protein_ids):
                    current_seq.append(line)
        
        if current_id and current_seq:
            if protein_ids is None or current_id in protein_ids:
                sequences[current_id] = ''.join(current_seq)
    
    return sequences


def parse_mutation(mutation: str) -> Tuple[str, int, str]:
    """Parse mutation string like 'M1A' into (wt_aa, position, mut_aa)"""
    if len(mutation) < 3:
        raise ValueError(f"Invalid mutation format: {mutation}")
    wt_aa = mutation[0]
    mut_aa = mutation[-1]
    position = int(mutation[1:-1])
    return wt_aa, position, mut_aa


def validate_mutation(protein_id: str, mutation: str, sequences: Dict[str, str]) -> Tuple[bool, str]:
    """Validate that mutation matches the UniProt sequence"""
    if protein_id not in sequences:
        return False, f"Protein {protein_id} not found in UniProt database"
    
    try:
        wt_aa, position, mut_aa = parse_mutation(mutation)
    except (ValueError, IndexError) as e:
        return False, f"Invalid mutation format: {mutation}"
    
    sequence = sequences[protein_id]
    
    # Check position is valid (1-indexed)
    if position < 1 or position > len(sequence):
        return False, f"Position {position} out of range for {protein_id} (length: {len(sequence)})"
    
    # Check wild-type amino acid matches
    actual_aa = sequence[position - 1]
    if actual_aa != wt_aa:
        return False, f"Wild-type mismatch at position {position}: expected {wt_aa}, found {actual_aa}"
    
    # Check mutant amino acid is valid
    if mut_aa not in AA_ALPHABET:
        return False, f"Invalid mutant amino acid: {mut_aa}"
    
    return True, "Valid"


def parse_input_file(file_path: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Parse input file and return list of (protein_id, mutation) tuples and full-scan proteins
    
    Supports:
    - TSV/TXT: space or tab-separated (P16501 M1A R56V)
    - CSV: comma-separated with optional header (Protein,Mutation or ID,Variation)
    
    Headers are auto-detected and skipped.
    
    If a protein has no mutation specified, it will be added to full_scan_proteins
    for all-mutations prediction.
    
    Returns:
        Tuple of (variants list, full_scan_proteins list)
    """
    variants = []
    full_scan_proteins = []
    
    # Detect file format
    is_csv = file_path.lower().endswith('.csv')
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return variants, full_scan_proteins
    
    # Check first line for header
    first_line = lines[0].strip().lower()
    has_header = False
    
    if is_csv:
        # CSV format: check for common header names
        if any(h in first_line for h in ['protein', 'uniprot', 'id', 'mutation', 'variation', 'variant']):
            has_header = True
    else:
        # TSV/TXT format: check if first line looks like a header
        parts = lines[0].strip().split()
        if len(parts) >= 1:
            # Check if it looks like a header
            first_part = parts[0].lower()
            if first_part in ['protein', 'uniprot', 'id', 'gene']:
                has_header = True
    
    start_idx = 1 if has_header else 0
    
    for line in lines[start_idx:]:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Parse based on format
        if is_csv:
            parts = [p.strip() for p in line.split(',')]
        else:
            parts = line.split()
        
        if len(parts) < 1:
            continue
        
        protein_id = parts[0]
        mutations = parts[1:] if len(parts) > 1 else []
        
        # Filter out empty mutations
        mutations = [m for m in mutations if m]
        
        if not mutations:
            # No mutation specified -> full scan for this protein
            full_scan_proteins.append(protein_id)
        else:
            for mutation in mutations:
                variants.append((protein_id, mutation))
    
    return variants, full_scan_proteins


def generate_all_mutations(protein_id: str, sequence: str) -> List[Tuple[str, str]]:
    """Generate all possible single amino acid mutations for a protein
    
    For each position, generates 19 mutations (all AAs except the wild-type)
    
    Args:
        protein_id: UniProt ID
        sequence: Protein sequence
    
    Returns:
        List of (protein_id, mutation) tuples
    """
    variants = []
    
    for pos, wt_aa in enumerate(sequence, 1):
        # Skip non-standard amino acids
        if wt_aa not in AA_ALPHABET:
            continue
        
        # Generate all possible mutations at this position
        for mut_aa in AA_ALPHABET:
            if mut_aa != wt_aa:
                mutation = f"{wt_aa}{pos}{mut_aa}"
                variants.append((protein_id, mutation))
    
    return variants


def query_database(db_path: str, variants: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Optional[float]]:
    """Query database for existing predictions using batch queries"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    results = {v: None for v in variants}  # Initialize all as None
    
    # Query in batches to avoid SQLite limits
    batch_size = 500
    for i in range(0, len(variants), batch_size):
        batch = variants[i:i + batch_size]
        
        # Build batch query
        placeholders = ','.join(['(?, ?)'] * len(batch))
        flat_params = [item for pair in batch for item in pair]
        
        query = f"SELECT protein_id, mutation, score FROM mutations WHERE (protein_id, mutation) IN (VALUES {placeholders})"
        cursor.execute(query, flat_params)
        
        for row in cursor.fetchall():
            protein_id, mutation, score = row
            results[(protein_id, mutation)] = score
    
    conn.close()
    return results


def load_plm_model(model_name: str, device: torch.device):
    """Load protein language model"""
    try:
        if model_name == "ankh2_large":
            print(f"        Loading Ankh2 Large...")
            model = T5EncoderModel.from_pretrained(
                "ElnaggarLab/ankh2-ext2",
                revision="4c155ee6b1aeb7f29ebf06a0399b331504104b67",
                torch_dtype=torch.float32
            ).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(
                "ElnaggarLab/ankh2-ext2",
                revision="4c155ee6b1aeb7f29ebf06a0399b331504104b67"
            )
        elif model_name == "esmc_600m":
            print(f"        Loading ESMC 600M...")
            model = AutoModelForMaskedLM.from_pretrained(
                "Synthyra/ESMplusplus_large",
                revision="1408244c8c08fb1b593da75b912b56a1688be86e",
                trust_remote_code=True,
                torch_dtype=torch.float32
            ).to(device).eval()
            tokenizer = model.tokenizer
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load {model_name}: {e}")


def truncate_sequence_for_mutation(sequence: str, mutation: str, window_size: int = 1024, max_length: int = 1024) -> Tuple[str, int, int]:
    """Truncate sequence to window around mutation if longer than max_length
    
    Returns:
        Tuple of (truncated_sequence, adjusted_position, start_position)
        start_position is the 0-based index where truncation starts in the original sequence
    """
    res_wt = mutation[0]
    pos = int(mutation[1:-1])
    res_mut = mutation[-1]
    
    len_seq = len(sequence)
    seq_pos = pos - 1  # Convert to 0-based index
    
    # Validate position
    if seq_pos < 0 or seq_pos >= len_seq:
        raise ValueError(f"Invalid position {pos} for sequence of length {len_seq}")
    
    # Validate wild-type residue
    if sequence[seq_pos] != res_wt:
        raise ValueError(f"Residue mismatch at position {pos}: expected '{res_wt}', found '{sequence[seq_pos]}'")
    
    # No truncation needed if sequence is short enough
    if len_seq <= max_length:
        return sequence, pos, 0
    
    # Calculate window boundaries centered on mutation
    half_window = window_size // 2
    start = seq_pos - half_window
    end = seq_pos + half_window
    
    # Adjust window if mutation is near the start
    if start < 0:
        start = 0
        end = window_size
    # Adjust window if mutation is near the end
    elif end > len_seq:
        end = len_seq
        start = len_seq - window_size
    
    # Ensure window does not exceed sequence boundaries
    start = max(start, 0)
    end = min(end, len_seq)
    
    # Truncate the sequence
    truncated_sequence = sequence[start:end]
    
    # Calculate adjusted position in truncated sequence (1-based)
    adjusted_position = seq_pos - start + 1
    
    return truncated_sequence, adjusted_position, start


def generate_embeddings_for_variants(
    variants_to_process: List[Tuple[str, str]],
    sequences: Dict[str, str],
    model_name: str,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Generate embeddings for wild-type and mutant sequences"""
    model, tokenizer = load_plm_model(model_name, device)
    
    all_embeddings = {}
    
    with torch.no_grad():
        for protein_id, mutation in tqdm(variants_to_process, 
                                          desc=f"    {model_name}",
                                          dynamic_ncols=True):
            sequence = sequences.get(protein_id)
            if not sequence:
                continue
            
            try:
                wt_aa, position, mut_aa = parse_mutation(mutation)
                
                # Truncate WT sequence and get the window position
                wt_seq_trunc, adjusted_pos, start_pos = truncate_sequence_for_mutation(sequence, mutation)
                
                # Create mutant sequence from FULL sequence
                mut_seq_full = sequence[:position-1] + mut_aa + sequence[position:]
                
                # Truncate mutant sequence using the SAME window as WT
                if len(sequence) > 1024:
                    end_pos = min(start_pos + len(wt_seq_trunc), len(mut_seq_full))
                    mut_seq_trunc = mut_seq_full[start_pos:end_pos]
                else:
                    mut_seq_trunc = mut_seq_full
                
                # Generate embeddings on truncated sequences
                wt_emb = embed_sequence(wt_seq_trunc, model, tokenizer, model_name, device)
                mut_emb = embed_sequence(mut_seq_trunc, model, tokenizer, model_name, device)
                
                # Store embeddings using adjusted position
                wt_emb_at_pos = wt_emb[adjusted_pos - 1]
                mut_emb_at_pos = mut_emb[adjusted_pos - 1]
                
                all_embeddings[f"{protein_id}_{mutation}_wt_emb"] = wt_emb_at_pos
                all_embeddings[f"{protein_id}_{mutation}_mut_emb"] = mut_emb_at_pos
                all_embeddings[f"{protein_id}_{mutation}_position"] = position
                all_embeddings[f"{protein_id}_{mutation}_adjusted_position"] = adjusted_pos
                
            except Exception as e:
                continue
    
    return all_embeddings


def embed_sequence(sequence: str, model, tokenizer, model_name: str, device: torch.device) -> torch.Tensor:
    """Generate embedding for a single sequence"""
    if model_name == "ankh2_large":
        sequences = [list(sequence)]
        outputs = tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt"
        )
        input_ids = outputs['input_ids'].to(device)
        attention_mask = outputs['attention_mask'].to(device)
        
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        seq_len = (attention_mask[0] == 1).sum()
        clean_emb = embeddings[0][:seq_len-1]
        
        return clean_emb.float().cpu()
    
    elif model_name == "esmc_600m":
        sequences = [list(sequence)]
        outputs = tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt"
        ).to(device)
        
        embeddings = model(**outputs).last_hidden_state
        attention_mask = outputs['attention_mask']
        seq_len = attention_mask[0].sum().item()
        clean_emb = embeddings[0][1:seq_len]
        
        return clean_emb.float().cpu()


def load_pathos_model(model_name: str, plm_type: str, device: torch.device, model_path: str = None) -> nn.Module:
    """Load trained PATHOS model weights
    
    Args:
        model_name: Name identifier for the model
        plm_type: Type of PLM (esmc_600m or ankh2_large)
        device: torch device
        model_path: Path to models folder
    
    Returns:
        Loaded model in evaluation mode
    """
    emb_dim = PLM_EMBEDDING_DIMS[plm_type]
    input_size = emb_dim * 2 + 18 + 3  # mut_emb + wt_emb + features
    
    model = FC_model(input_size=input_size).to(device)
    
    # Load trained weights
    if model_path is None:
        model_path = MODELS_FOLDER
    
    checkpoint_file = TRAINED_MODELS[plm_type]
    checkpoint_path = os.path.join(model_path, checkpoint_file)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def run_pathos_inference(
    variants_to_process: List[Tuple[str, str]],
    embeddings_esmc: Dict[str, torch.Tensor],
    embeddings_ankh: Dict[str, torch.Tensor],
    features_df: pd.DataFrame,
    device: torch.device,
    model_path: str = None
) -> Dict[Tuple[str, str], float]:
    """Run PATHOS inference using both PLM models with actual features"""
    predictions = {}
    
    # Load models once
    models = {}
    for plm_name in PLM_MODELS:
        models[plm_name] = load_pathos_model(f"pathos_{plm_name}", plm_name, device, model_path)
    
    # Create lookup for features
    features_lookup = {}
    for _, row in features_df.iterrows():
        key = (row['ID'], row['Variation'])
        features_lookup[key] = {
            'PASTML': row['PASTML'],
            'AF': row['AF'],
            'STRING': row['STRING'],
            'ANNOTATIONS': row['ANNOTATIONS']
        }
    
    # Process variants
    for protein_id, mutation in tqdm(variants_to_process, desc="Inference", dynamic_ncols=True):
        scores = []
        
        feat_key = (protein_id, mutation)
        if feat_key not in features_lookup:
            continue
        
        feat_data = features_lookup[feat_key]
        
        for plm_name in PLM_MODELS:
            if plm_name == "esmc_600m":
                embeddings = embeddings_esmc
            else:
                embeddings = embeddings_ankh
            
            mut_key = f"{protein_id}_{mutation}_mut_emb"
            wt_key = f"{protein_id}_{mutation}_wt_emb"
            
            if mut_key not in embeddings or wt_key not in embeddings:
                continue
            
            mut_emb = embeddings[mut_key]
            wt_emb = embeddings[wt_key]
            
            # Build feature vector
            features = torch.tensor([
                feat_data['PASTML'],
                feat_data['AF'],
                feat_data['STRING']
            ] + feat_data['ANNOTATIONS'], dtype=torch.float32)
            
            input_features = torch.cat([mut_emb, wt_emb, features]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                score = models[plm_name](input_features).item()
                scores.append(score)
        
        if scores:
            predictions[(protein_id, mutation)] = np.mean(scores)
    
    return predictions


def process_variants_batched(
    variants_to_process: List[Tuple[str, str]],
    sequences: Dict[str, str],
    features_df: pd.DataFrame,
    device: torch.device,
    model_path: str = None,
    batch_size: int = 100
) -> Dict[Tuple[str, str], float]:
    """Process variants in batches to avoid memory overload
    
    Optimizations:
    - ESMC runs on CPU while Ankh2 runs on GPU in parallel
    - WT embeddings cached per (protein, truncation_window) to avoid recomputation
    - PLM models loaded once (ESMC on CPU, Ankh2 on GPU)
    
    Args:
        variants_to_process: List of (protein_id, mutation) tuples
        sequences: Dictionary mapping protein IDs to sequences
        features_df: DataFrame with precomputed features
        device: torch device (used for Ankh2 and PATHOS inference)
        model_path: Path to PATHOS models
        batch_size: Number of variants per batch (default: 100)
    
    Returns:
        Dictionary mapping (protein_id, mutation) to prediction score
    """
    import gc
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    all_predictions = {}
    n_batches = (len(variants_to_process) + batch_size - 1) // batch_size
    
    # Create features lookup once using set_index for O(1) lookup
    features_lookup = features_df.set_index(['ID', 'Variation']).to_dict('index')
    
    # Define devices: ESMC on CPU, Ankh2 on GPU
    cpu_device = torch.device('cpu')
    gpu_device = device if torch.cuda.is_available() else cpu_device
    
    # Load PATHOS models (they're small, run inference on GPU)
    pathos_models = {}
    for plm_name in PLM_MODELS:
        pathos_models[plm_name] = load_pathos_model(f"pathos_{plm_name}", plm_name, gpu_device, model_path)
    
    # Load PLM models: ESMC on CPU, Ankh2 on GPU
    print(f"    Loading esmc_600m on CPU...")
    esmc_model, esmc_tokenizer = load_plm_model("esmc_600m", cpu_device)
    
    print(f"    Loading ankh2_large on GPU...")
    ankh_model, ankh_tokenizer = load_plm_model("ankh2_large", gpu_device)
    
    plm_models = {
        "esmc_600m": (esmc_model, esmc_tokenizer, cpu_device),
        "ankh2_large": (ankh_model, ankh_tokenizer, gpu_device)
    }
    
    # WT cache per PLM with thread lock for safety
    wt_cache = {}
    wt_cache_lock = threading.Lock()
    
    # Progress tracking per PLM
    progress_counters = {"esmc_600m": 0, "ankh2_large": 0}
    progress_lock = threading.Lock()
    
    print(f"    Processing {len(variants_to_process)} variants in {n_batches} batches of {batch_size}")
    print(f"    Mode: ESMC (CPU) + Ankh2 (GPU) in parallel\n")
    
    def generate_embeddings_for_plm(plm_name, batch_variants, model, tokenizer, plm_device):
        """Generate embeddings for a specific PLM on its designated device"""
        embeddings = {}
        device_str = "CPU" if plm_device.type == 'cpu' else "GPU"
        
        start_time = time.time()
        print(f"      [{plm_name}] Started on {device_str} ({len(batch_variants)} variants)")
        
        with torch.no_grad():
            for i, (protein_id, mutation) in enumerate(batch_variants):
                sequence = sequences.get(protein_id)
                if not sequence:
                    continue
                
                try:
                    wt_aa, position, mut_aa = parse_mutation(mutation)
                    
                    # Truncate sequences
                    wt_seq_trunc, adjusted_pos, start_pos = truncate_sequence_for_mutation(sequence, mutation)
                    
                    # Cache key for WT embedding (per PLM)
                    cache_key = (plm_name, protein_id, start_pos, len(wt_seq_trunc))
                    
                    # Get or compute WT embedding (thread-safe)
                    with wt_cache_lock:
                        if cache_key not in wt_cache:
                            wt_emb_full = embed_sequence(wt_seq_trunc, model, tokenizer, plm_name, plm_device)
                            wt_cache[cache_key] = wt_emb_full.cpu()
                        wt_emb_full = wt_cache[cache_key]
                    
                    wt_emb_at_pos = wt_emb_full[adjusted_pos - 1]
                    
                    # Generate mutant sequence and embedding
                    mut_seq_full = sequence[:position-1] + mut_aa + sequence[position:]
                    if len(sequence) > 1024:
                        end_pos = min(start_pos + len(wt_seq_trunc), len(mut_seq_full))
                        mut_seq_trunc = mut_seq_full[start_pos:end_pos]
                    else:
                        mut_seq_trunc = mut_seq_full
                    
                    mut_emb = embed_sequence(mut_seq_trunc, model, tokenizer, plm_name, plm_device)
                    mut_emb_at_pos = mut_emb[adjusted_pos - 1].cpu()
                    
                    embeddings[f"{protein_id}_{mutation}_wt"] = wt_emb_at_pos
                    embeddings[f"{protein_id}_{mutation}_mut"] = mut_emb_at_pos
                    
                    del mut_emb
                    
                    # Update progress counter
                    with progress_lock:
                        progress_counters[plm_name] = i + 1
                    
                except Exception as e:
                    continue
        
        elapsed = time.time() - start_time
        print(f"      [{plm_name}] Finished on {device_str} in {elapsed:.1f}s ({len(embeddings)//2} variants)")
        
        return plm_name, embeddings
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(variants_to_process))
        batch_variants = variants_to_process[start_idx:end_idx]
        
        print(f"    Batch {batch_idx + 1}/{n_batches} ({len(batch_variants)} variants)")
        
        # Reset progress counters
        progress_counters["esmc_600m"] = 0
        progress_counters["ankh2_large"] = 0
        
        # Run ESMC (CPU) and Ankh2 (GPU) in parallel
        batch_embeddings = {}
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for plm_name in PLM_MODELS:
                model, tokenizer, plm_device = plm_models[plm_name]
                future = executor.submit(
                    generate_embeddings_for_plm,
                    plm_name, batch_variants, model, tokenizer, plm_device
                )
                futures.append(future)
            
            # Wait for both to complete
            for future in as_completed(futures):
                plm_name, embeddings = future.result()
                batch_embeddings[plm_name] = embeddings
        
        print(f"      Both PLMs completed, running PATHOS inference...")
        
        # Run PATHOS inference for this batch (on GPU, average both PLMs)
        for protein_id, mutation in batch_variants:
            feat_key = (protein_id, mutation)
            
            if feat_key not in features_lookup:
                continue
            
            feat_data = features_lookup[feat_key]
            scores = []
            
            for plm_name in PLM_MODELS:
                wt_key = f"{protein_id}_{mutation}_wt"
                mut_key = f"{protein_id}_{mutation}_mut"
                
                if wt_key not in batch_embeddings[plm_name] or mut_key not in batch_embeddings[plm_name]:
                    continue
                
                wt_emb = batch_embeddings[plm_name][wt_key]
                mut_emb = batch_embeddings[plm_name][mut_key]
                
                features = torch.tensor([
                    feat_data['PASTML'],
                    feat_data['AF'],
                    feat_data['STRING']
                ] + feat_data['ANNOTATIONS'], dtype=torch.float32)
                
                input_features = torch.cat([mut_emb, wt_emb, features]).unsqueeze(0).to(gpu_device)
                
                with torch.no_grad():
                    score = pathos_models[plm_name](input_features).item()
                scores.append(score)
            
            if scores:
                all_predictions[(protein_id, mutation)] = np.mean(scores)
        
        # Clear batch embeddings
        del batch_embeddings
        gc.collect()
    
    # Cleanup: unload PLM models and clear cache
    del plm_models, wt_cache, esmc_model, ankh_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return all_predictions


def save_results(
    variants: List[Tuple[str, str]],
    db_results: Dict[Tuple[str, str], Optional[float]],
    new_predictions: Dict[Tuple[str, str], float],
    output_path: str
):
    """Save combined results to CSV"""
    rows = []
    for protein_id, mutation in variants:
        if db_results[(protein_id, mutation)] is not None:
            score = db_results[(protein_id, mutation)]
            source = "database"
        elif (protein_id, mutation) in new_predictions:
            score = new_predictions[(protein_id, mutation)]
            source = "predicted"
        else:
            score = None
            source = "failed"
        
        interpretation = "Pathogenic" if score and score >= 0.63 else "Benign" if score else "N/A"
        
        rows.append({
            "Protein": protein_id,
            "Mutation": mutation,
            "PATHOS_score": f"{score:.4f}" if score else "N/A",
            "Interpretation": interpretation,
            "Source": source
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run PATHOS predictions for protein variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pathos.py --input variants.txt --output results.csv
  python run_pathos.py -i variants.txt -o results.csv
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input file with protein IDs and mutations"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output CSV file for results"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers for feature generation (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation in batched mode (default: 100)"
    )
    parser.add_argument(
        "--mmseqs-mem-limit",
        default="8G",
        help="Memory limit for mmseqs2 MSA generation (default: 8G)"
    )
    parser.add_argument(
        "--batch-threshold",
        type=int,
        default=10000,
        help="Number of variants above which batched mode is enabled to save memory (default: 10000, use 0 to always batch)"
    )
    
    args = parser.parse_args()
    
    # Validate required files exist
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found: {DB_PATH}")
        print(f"       Run setup_pathos.sh to download the database.")
        sys.exit(1)
    
    if not os.path.exists(FASTA_PATH):
        print(f"ERROR: FASTA file not found: {FASTA_PATH}")
        print(f"       Run setup_pathos.sh to download the database.")
        sys.exit(1)
    
    if not os.path.exists(MODELS_FOLDER):
        print(f"ERROR: Models folder not found: {MODELS_FOLDER}")
        print(f"       Run setup_pathos.sh to download the models.")
        sys.exit(1)
    
    if not os.path.exists(AF_SQLITE_PATH):
        print(f"ERROR: Allele frequency database not found: {AF_SQLITE_PATH}")
        print(f"       Run setup_pathos.sh to download the database.")
        sys.exit(1)
    
    total_steps = 8
    start_time = time.time()
    
    # Step 1: Parse input file
    print(f"\n[1/{total_steps}] Parsing input file...")
    variants, full_scan_proteins = parse_input_file(args.input)
    
    # Handle full-scan proteins (no mutation specified -> generate all mutations)
    if full_scan_proteins:
        print(f"    Found {len(full_scan_proteins)} proteins for full scan (all mutations)")
        print(f"    Loading sequences to generate all possible mutations...")
        scan_sequences = load_fasta_sequences(FASTA_PATH, protein_ids=full_scan_proteins)
        
        full_scan_variants = []
        for protein_id in full_scan_proteins:
            if protein_id in scan_sequences:
                seq = scan_sequences[protein_id]
                protein_mutations = generate_all_mutations(protein_id, seq)
                full_scan_variants.extend(protein_mutations)
                print(f"    {protein_id}: {len(seq)} residues -> {len(protein_mutations)} mutations")
            else:
                print(f"    WARNING: {protein_id} not found in FASTA, skipping")
        
        print(f"    Total full-scan mutations: {len(full_scan_variants)}")
        # Append full-scan variants to the main list (will be processed after regular variants)
        variants.extend(full_scan_variants)
    
    unique_proteins = list(set([protein_id for protein_id, _ in variants]))
    print(f"    Total: {len(variants)} variants across {len(unique_proteins)} proteins")
    
    # Step 2: Query database for existing predictions (no sequences needed)
    print(f"\n[2/{total_steps}] Querying PATHOS database for existing predictions...")
    db_results = query_database(DB_PATH, variants)
    variants_in_db = [(p, m) for (p, m) in variants if db_results[(p, m)] is not None]
    variants_to_process = [(p, m) for (p, m) in variants if db_results[(p, m)] is None]
    print(f"    {len(variants_in_db)} found in database, {len(variants_to_process)} need de novo prediction")
    
    new_predictions = {}
    valid_variants = list(variants)  # Start with all variants
    
    if variants_to_process:
        # Step 3: Load UniProt sequences
        proteins_to_load = list(set([p for p, m in variants_to_process]))
        print(f"\n[3/{total_steps}] Loading UniProt sequences for {len(proteins_to_load)} proteins...")
        sequences = load_fasta_sequences(FASTA_PATH, protein_ids=proteins_to_load)
        print(f"    Loaded {len(sequences)} sequences from FASTA")
        
        # Step 4: Validate mutations against sequences
        print(f"\n[4/{total_steps}] Validating mutations against UniProt sequences...")
        validated_variants = []
        invalid_count = 0
        for protein_id, mutation in variants_to_process:
            is_valid, _ = validate_mutation(protein_id, mutation, sequences)
            if is_valid:
                validated_variants.append((protein_id, mutation))
            else:
                invalid_count += 1
        print(f"    {len(validated_variants)} valid, {invalid_count} invalid (sequence mismatch)")
        
        if not validated_variants:
            print("WARNING: No valid variants to predict de novo.")
        else:
            variants_to_process = validated_variants
            
            # Check and generate MSAs for all proteins (PASTML is always enabled)
            proteins_needing_msa = list(set([p for p, _ in variants_to_process]))
            print(f"\n[5/{total_steps}] Checking MSA availability for {len(proteins_needing_msa)} proteins...")
            
            available_msas, generated_msas, failed_msas = check_and_generate_msas(
                proteins_needing_msa,
                MSA_FOLDER,
                FASTA_FOLDER,
                MAMMALS_DB,
                mem_limit=args.mmseqs_mem_limit,
                debug=False
            )
            
            if generated_msas:
                print(f"    Successfully generated: {len(generated_msas)}")
            if failed_msas:
                print(f"    Failed to generate: {len(failed_msas)} (PASTML will return NaN for these)")
                
                feature_step = 6
            else:
                feature_step = 5
            
            # Generate features
            n_workers = args.n_jobs if args.n_jobs else max(1, multiprocessing.cpu_count() - 1)
            print(f"\n[{feature_step}/{total_steps}] Generating features for {len(variants_to_process)} variants ({n_workers} workers)...")
            print(f"    - PASTML: computing conservation scores")
            print(f"    - AF: querying allele frequencies from gnomAD")
            print(f"    - STRING: loading protein interaction scores")
            print(f"    - UniProt: extracting 18 annotation features (GFF)")
            features_df = generate_features_for_variants(
                variants_to_process, 
                sequences,
                os.path.dirname(FASTA_PATH),
                tree_path=TREE_PATH,
                msa_folder=MSA_FOLDER,
                pastml_cache=PASTML_CACHE,
                compute_pastml=True,
                af_sqlite=AF_SQLITE_PATH,
                n_jobs=args.n_jobs,
                fasta_folder=FASTA_FOLDER,
                mammals_db=MAMMALS_DB
            )
            
            # Step 7-8: Generate embeddings and run inference
            # Use batched mode for large datasets to save memory
            emb_step = feature_step + 1
            inf_step = feature_step + 2
            
            if len(variants_to_process) > args.batch_threshold:
                print(f"\n[{emb_step}-{inf_step}/{total_steps}] Generating embeddings and running inference (batched mode for {len(variants_to_process)} variants)...")
                new_predictions = process_variants_batched(
                    variants_to_process,
                    sequences,
                    features_df,
                    DEVICE,
                    MODELS_FOLDER,
                    batch_size=args.batch_size
                )
            else:
                # Standard mode: generate all embeddings sequentially, then run inference
                print(f"\n[{emb_step}/{total_steps}] Generating WT and mutant embeddings with PLMs...")
                print(f"    ESMC 600M:")
                embeddings_esmc = generate_embeddings_for_variants(
                    variants_to_process, sequences, "esmc_600m", DEVICE
                )
                print(f"    Ankh2 Large:")
                embeddings_ankh = generate_embeddings_for_variants(
                    variants_to_process, sequences, "ankh2_large", DEVICE
                )
                
                print(f"\n[{inf_step}/{total_steps}] Running PATHOS inference...")
                new_predictions = run_pathos_inference(
                    variants_to_process,
                    embeddings_esmc,
                    embeddings_ankh,
                    features_df,
                    DEVICE,
                    MODELS_FOLDER
                )
    else:
        print(f"\n[3-8/{total_steps}] All variants found in database, skipping prediction steps.")
    
    # Update valid_variants to include only those we have results for
    valid_variants = variants_in_db + list(new_predictions.keys())
    
    # Save results
    save_results(variants, db_results, new_predictions, args.output)
    
    # Final summary
    total_time = time.time() - start_time
    pathogenic = sum(1 for k, v in new_predictions.items() if v >= 0.63)
    benign = len(new_predictions) - pathogenic
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"  From database: {len(variants_in_db)}")
    print(f"  Newly predicted: {len(new_predictions)} ({pathogenic} pathogenic, {benign} benign)")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*60}")
    
    # Print final CSV
    print(f"\n{pd.read_csv(args.output).head(50).to_string(index=False)}")


if __name__ == "__main__":
    main()
