"""
Geographical Distance Matrix Generator

[Function Description]
This script calculates the Euclidean geographical distance matrix between samples
based on columns 3 and 4 of the dataset, and saves the result as a CSV file
for use by the main model script.

[File Path Instructions]
- Input dataset: stored in ../Data/ directory, with the following format:
  The first 4 columns are coordinate variables,
  column 5 is the dependent variable,
  columns 6 and beyond are independent variables.
- Output matrix: saved to ../Data/Geographical_distance_matrix.csv.

[Notes]
- Coordinates are read from columns 3 and 4.
- The output CSV file includes column names (V1, V2, ...) and does not include row indices.
"""
import pandas as pd
import numpy as np
import os
from scipy.spatial import distance_matrix

# ====================== User Configuration ======================
# Path to the raw dataset
main_file = '../Data/your_data.csv'   # Example: Housing.csv, zn.csv, Crime.csv, etc.
# Path to the output distance matrix
output_file = '../Data/Geographical_distance_matrix.csv'
# ===============================================================

def load_data_for_geo(file_path):
    """Read the data and extract coordinates from columns 3 and 4"""
    data = pd.read_csv(file_path)
    print(f"Reading dataset: {file_path}, data shape: {data.shape}")
    coords = data.iloc[:, 2:4].values
    print(f"Coordinate matrix shape: {coords.shape}")
    return coords, data

def compute_geo_matrix(coords):
    """Compute the Euclidean geographical distance matrix"""
    n = coords.shape[0]
    dist_matrix = distance_matrix(coords, coords)
    return dist_matrix, n

def main():
    coords, data = load_data_for_geo(main_file)
    n = coords.shape[0]
    print(f"Total number of samples: {n}")

    print("Computing geographical distance matrix...")
    dist_matrix, n = compute_geo_matrix(coords)

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    column_names = [f'V{i+1}' for i in range(n)]
    pd.DataFrame(dist_matrix).to_csv(output_file, index=False, header=column_names)
    print(f"Distance matrix saved to: {output_file}")

if __name__ == "__main__":
    main()