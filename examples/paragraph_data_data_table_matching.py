import csv

csv_source_path = "./jingjun/figures.csv"
csv_path = "./jingjun/paragraph_figure_final.csv"
csv_path_new = "./jingjun/paragraph_figure_more_enhanced.csv"

# First, read the source CSV and create a lookup dictionary
print("Loading source figure data...")
figure_lookup = {}
with open(csv_source_path, 'r', newline='', encoding='utf-8') as source_file:
    source_reader = csv.DictReader(source_file)
    for row in source_reader:
        paper_arxiv_id = row['paper_arxiv_id']
        label = row['label']
        # Create a composite key for lookup
        key = (paper_arxiv_id, label)
        figure_lookup[key] = {
            'figure_id': row['id'],
            'caption': row['caption']
        }

print(f"Loaded {len(figure_lookup)} figure records from source file")

# Now process the main CSV file and enhance it
print("Processing and enhancing main CSV file...")
with open(csv_path, 'r', newline='', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)
    
    # Get the original fieldnames and add new ones
    original_fieldnames = reader.fieldnames
    new_fieldnames = original_fieldnames + ['figure_id', 'caption']
    
    # Open the new CSV file for writing
    with open(csv_path_new, 'w', newline='', encoding='utf-8') as csv_file_new:
        writer = csv.DictWriter(csv_file_new, fieldnames=new_fieldnames)
        writer.writeheader()
        
        processed_count = 0
        matched_count = 0
        
        for line in reader:
            paper_arxiv_id = line['paper_arxiv_id']
            reference_label = line['reference_label']
            label = f"\\label{{{reference_label}}}"
            
            # Create lookup key
            lookup_key = (paper_arxiv_id, label)
            
            # Look up the figure information in our dictionary
            if lookup_key in figure_lookup:
                figure_info = figure_lookup[lookup_key]
                line['figure_id'] = figure_info['figure_id']
                line['caption'] = figure_info['caption']
                matched_count += 1
            else:
                # Handle cases where no matching figure is found
                line['figure_id'] = None
                line['caption'] = None
            
            # Write the enhanced row to the new CSV
            writer.writerow(line)
            processed_count += 1

print(f"Processing complete!")
print(f"Processed {processed_count} rows")
print(f"Matched {matched_count} rows with figure data")
print(f"Enhanced data saved to {csv_path_new}")