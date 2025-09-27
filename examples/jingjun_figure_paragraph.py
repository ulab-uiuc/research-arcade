import psycopg2
from psycopg2.extras import Json
import json
import csv
import argparse

# Assuming you're using argparse for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dest_csv_path', required=True, help='Destination CSV file path')
parser.add_argument('--file_path', required=True, help='Input JSON file path')
args = parser.parse_args()

dest_csv_path = args.dest_csv_path
file_path = args.file_path

conn = psycopg2.connect(
    host="localhost",
    port="5433",
    dbname="postgres",
    user="cl195"
)

conn.autocommit = True
cur = conn.cursor()

with open(file_path, "r") as f:
    json_file = json.load(f)

# Collect arxiv IDs
arxiv_ids = list(json_file)

with open(dest_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)  # Fixed: need to create csv.writer object
    csv_writer.writerow(['id', 'paragraph_id', 'paper_section', 'paper_arxiv_id', 'reference_label', 'reference_type', 'global_paragraph_id', 'figure_id'])
    
    row_id = 1  # Counter for row IDs
    
    for arxiv_id in arxiv_ids:
        # Search for paragraph_figures
        statement = "SELECT * FROM paragraph_references WHERE reference_type = 'figure' AND paper_arxiv_id = %s;"
        cur.execute(statement, (arxiv_id,))
        
        para_figures = cur.fetchall()
        
        # Find two things: One is the exact paragraph, the other is the exact figure
        for para_figure in para_figures:
            # Assuming the structure of paragraph_references table
            # Adjust indices based on your actual table structure
            paragraph_id = para_figure[1]
            paper_section = para_figure[2]
            reference_label = para_figure[4]  # Assuming this is the actual figure reference
            reference_type = 'figure'  # Since we're filtering by reference_label = 'figure'
            
            label = f"\\label{{{reference_label}}}"  # Fixed: escaped backslash
            
            # First search for the global paragraph id
            statement2 = "SELECT id FROM paragraphs WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s;"  # Fixed SQL syntax
            cur.execute(statement2, (arxiv_id, paper_section, paragraph_id))
            
            global_paragraph_result = cur.fetchone()
            global_paragraph_id = global_paragraph_result[0] if global_paragraph_result else None
            
            # Then we fetch figure id
            statement3 = "SELECT id FROM figures WHERE label = %s AND paper_arxiv_id = %s;"
            cur.execute(statement3, (label, arxiv_id))  # Fixed: use arxiv_id instead of undefined processed_arxiv_id
            
            figure_result = cur.fetchone()
            figure_id = figure_result[0] if figure_result else None
            
            # Save this line to csv file
            row = [
                row_id,
                paragraph_id,
                paper_section,
                arxiv_id,
                reference_label,
                reference_type,
                global_paragraph_id,
                figure_id
            ]
            csv_writer.writerow(row)
            row_id += 1

# Close database connection
cur.close()
conn.close()

print(f"Data extraction completed. Results saved to {dest_csv_path}")