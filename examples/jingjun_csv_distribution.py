import csv

csv_file_path = "./csv/arxiv_old_in_figure2.csv"  # Fixed extension
n = 2

# Open output files
output_files = []
writers = []
for i in range(6, 8):
    file = open(f"./csv/arxiv_old_in_figure_{i}.csv", 'w', newline='')
    output_files.append(file)
    writers.append(csv.writer(file))

# Read and distribute rows
with open(csv_file_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    header = next(reader)  # Read header
    
    # Write header to all files
    for writer in writers:
        writer.writerow(header)
    
    # First pass: count total rows
    rows = list(reader)
    total_rows = len(rows)
    
    # Calculate chunk sizes for sequential distribution
    chunk_size = total_rows // n
    remainder = total_rows % n
    
    # Distribute rows sequentially
    start_idx = 0
    for i in range(n):
        # Add 1 extra row to first 'remainder' files to handle uneven division
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        for row in rows[start_idx:end_idx]:
            writers[i].writerow(row)
        
        start_idx = end_idx

# Close all output files
for file in output_files:
    file.close()

print(f"CSV distributed into {n} files")