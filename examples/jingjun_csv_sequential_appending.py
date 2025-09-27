import csv
import os

dest_path = "./jingjun/paragraph_figure_enhanced_preprocessed.csv"

# Track if we've written headers yet
headers_written = False

# Open destination file in append mode (change to 'w' if you want to overwrite)
with open(dest_path, 'a', newline='', encoding='utf-8') as dest_file:
    dest_writer = None
    
    for i in range(1, 6, 1):
        source_file = f"./jingjun/paragraph_figure_enhanced_{i}_preprocessed.csv"
        
        # Check if source file exists
        if not os.path.exists(source_file):
            print(f"Warning: {source_file} not found, skipping...")
            continue
        
        # Read each source file
        with open(source_file, 'r', newline='', encoding='utf-8') as src_file:
            src_reader = csv.reader(src_file)
            
            # Handle headers
            try:
                headers = next(src_reader)
                
                # Write headers only once (from first file)
                if not headers_written:
                    dest_writer = csv.writer(dest_file)
                    dest_writer.writerow(headers)
                    headers_written = True
                    print(f"Headers written from {source_file}")
                
                # Write data rows
                row_count = 0
                for row in src_reader:
                    dest_writer.writerow(row)
                    row_count += 1
                
                print(f"Merged {row_count} rows from {source_file}")
                
            except StopIteration:
                print(f"Warning: {source_file} is empty, skipping...")
                continue

print(f"All files merged into {dest_path}")