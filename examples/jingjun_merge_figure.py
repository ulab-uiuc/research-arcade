import csv
import json
import pandas as pd

# Load arxiv IDs for each year
arxiv_ids_list = []
dest_paths = []

for year in range(2021, 2025, 1):
    json_file = f"./json/arxiv_{year}.json"
    with open(json_file, 'r') as f:
        arxiv_ids = list(json.load(f))
        arxiv_ids_list.append(arxiv_ids)
        dest_paths.append(f"./csv/old_iclr/figure_preprocessed_{year}.csv")

# Process 7 CSV files
for i in range(1, 8):  # 1 to 7
    source_path = f"./csv/arxiv_old_in_figure_{i}_preprocessed.csv"
    
    # Read the CSV file
    df = pd.read_csv(source_path)
    
    # For each year (j corresponds to years 2021-2024)
    for j in range(0, 4, 1):
        # Filter rows where paper_arxiv_id is in the current year's arxiv_ids
        filtered_df = df[df['paper_arxiv_id'].isin(arxiv_ids_list[j])]
        
        # Append to destination file (create if doesn't exist)
        filtered_df.to_csv(dest_paths[j], mode='a', header=not pd.io.common.file_exists(dest_paths[j]), index=False)