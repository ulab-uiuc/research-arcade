import psycopg2
from psycopg2.extras import Json
import json
import argparse
import csv
import os

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dest_csv_path', required=True, help='Destination CSV file path')
parser.add_argument('--file_path', required=True, help='Input JSON file path of the arxiv ids')
args = parser.parse_args()

dest_csv_path = args.dest_csv_path
file_path = args.file_path

conn = psycopg2.connect(
    host="localhost",
    port="5433",
    dbname="postgres",
    user="cl195"
)
# Enable autocommit
conn.autocommit = True
cur = conn.cursor()

with open(file_path, "r") as f:
    json_file = json.load(f)

# Collect arxiv IDs
arxiv_ids = list(json_file)

# Prepare CSV file
with open(dest_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write header - adjust columns based on your paragraph table structure
    # Assuming paragraph table has: id, paragraph_id, content, paper_arxiv_id, paper_section, ...
    csv_writer.writerow(['id', 'paragraph_id', 'content', 'paper_arxiv_id', 'paper_section', 'section_order', 'paragraph_order'])

    for arxiv_id in arxiv_ids:
        # First find the sequence of sections 
        paper_file = f"./download/output/{arxiv_id}.json"
        section_order = {}
        paragraph_data = []
        
        # Check if paper file exists
        if not os.path.exists(paper_file):
            print(f"Warning: Paper file {paper_file} not found, skipping {arxiv_id}")
            continue
        
        try:
            # Load paper JSON to get section order
            with open(paper_file, 'r') as f:
                paper_json = json.load(f)
                
                # Check if 'sections' key exists
                if 'sections' not in paper_json:
                    print(f"Warning: No 'sections' key found in {paper_file}, skipping {arxiv_id}")
                    continue
                    
                sections = paper_json['sections']
                
                # Build section order mapping
                for i, (paper_section, _) in enumerate(sections.items(), 1):
                    section_order[paper_section] = i
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing paper file {paper_file}: {e}, skipping {arxiv_id}")
            continue

        # Select paragraphs for this arxiv_id
        statement = "SELECT * FROM paragraphs WHERE paper_arxiv_id = %s ORDER BY paragraph_id"
        cur.execute(statement, (arxiv_id,))
        
        paragraphs = cur.fetchall()
        
        if not paragraphs:
            print(f"No paragraphs found for arxiv_id: {arxiv_id}")
            continue
        
        # Process each paragraph
        for paragraph in paragraphs:
            # Assuming paragraph structure: [id, paragraph_id, content, paper_arxiv_id, paper_section, ...]
            paragraph_id = paragraph[1]
            paper_section = paragraph[4]
            
            # Get section order (default to 999 if section not found)
            sect_order = section_order.get(paper_section, 999)
            
            # Store paragraph with ordering info
            paragraph_data.append({
                'data': paragraph,
                'section_order': sect_order,
                'paragraph_id': paragraph_id
            })
        
        # Sort paragraphs based on:
        # 1. Section order (ascending - earlier sections first)
        # 2. Paragraph ID (ascending - earlier paragraphs first within same section)
        paragraph_data.sort(key=lambda x: (x['section_order'], x['paragraph_id']))
        
        # Write sorted paragraphs to CSV
        for j, item in enumerate(paragraph_data):
            paragraph = item['data']
            section_order_val = item['section_order']
            
            # Create row with original paragraph data plus ordering columns
            row = list(paragraph) + [section_order_val, j + 1]  # j+1 for 1-based paragraph order
            csv_writer.writerow(row)
        
        print(f"Processed {len(paragraph_data)} paragraphs for arxiv_id: {arxiv_id}")

# Close database connection
cur.close()
conn.close()

print(f"Paragraphs have been ordered and saved to {dest_csv_path}")