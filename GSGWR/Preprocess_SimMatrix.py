"""
Similarity Distance Matrix Generator

【Description】
This script calculates the attribute similarity distance matrix between samples in the dataset,
and saves the result as a CSV file for use by the main model script.

【File Paths】
- Input dataset: located under ../Data/, with the following format:
  First 4 columns = coordinate variables,
  Column 5 = dependent variable,
  Column 6 and beyond = independent variables.
- Output matrix: saved as ../Data/Similarity_distance_matrix.csv.

【Notes】
- Computation uses a chunking strategy (default block_size=500), adjustable according to memory.
- The output CSV file includes column names (V1, V2, ...), without row indices.
"""

import pandas as pd
import numpy as np
import os

# ====================== User Configuration ======================
# Path to the original dataset (located under ../Data/), modify according to actual file name
main_file = '../Data/your_data.csv'   # e.g., zn.csv, Crime.csv, HIV.csv, etc.
# Output distance matrix path (fixed name, matching the main script)
output_file = '../Data/Similarity_distance_matrix.csv'
# ===============================================================

def load_data_for_similarity(file_path):
    """Read data and extract independent variables"""
    data = pd.read_csv(file_path)
    print(f"Reading dataset: {file_path}, shape: {data.shape}")
    # First 4 columns: coordinates, 5th: dependent variable, 6th and beyond: independent variables
    X = data.iloc[:, 5:].values
    print(f"Independent variable matrix shape: {X.shape}")
    return X, data

def compute_similarity_matrix(X, block_size=500):
    """
    Compute attribute similarity matrix
    """
    n = X.shape[0]
    variances = np.var(X, axis=0)
    variances = np.maximum(variances, 1e-12)
    print("Variances of each variable:", variances)

    similarity_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        X_i = X[i:i_end, :]

        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            X_j = X[j:j_end, :]

            diffs = X_i[:, None, :] - X_j[None, :, :]
            terms = (diffs ** 4) / (2 * (variances ** 2))
            similarities_per_var = np.exp(-terms)
            similarities_block = np.min(similarities_per_var, axis=2)

            i_indices = np.arange(i, i_end)
            j_indices = np.arange(j, j_end)
            ii, jj = np.meshgrid(i_indices, j_indices, indexing='ij')
            mask = (ii != jj)
            similarity_matrix[ii[mask], jj[mask]] = similarities_block[mask]

        print(f"Processed {min(i + block_size, n)}/{n} points")

    np.fill_diagonal(similarity_matrix, 1.0)

    return similarity_matrix

def main():
    X, data = load_data_for_similarity(main_file)
    n = X.shape[0]
    print(f"Number of data points: {n}")

    print("Computing similarity matrix...")
    sim_matrix = compute_similarity_matrix(X, block_size=500)

    print("Converting similarity matrix to distance matrix (d = -ln(s + 1e-6))...")
    dist_matrix = -np.log(sim_matrix + 1e-6)

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    column_names = [f'V{i+1}' for i in range(n)]
    pd.DataFrame(dist_matrix).to_csv(output_file, index=False, header=column_names)
    print(f"Distance matrix saved to: {output_file}")

if __name__ == "__main__":
    main()
