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
    csv_writer = csv.writer(csvfile)
    # Updated header to reflect table-specific columns
    csv_writer.writerow(['id', 'paragraph_id', 'paper_section', 'paper_arxiv_id', 'reference_label', 'reference_type', 'global_paragraph_id', 'table_id'])
    
    row_id = 1  # Counter for row IDs
    
    for arxiv_id in arxiv_ids:
        # Search for paragraph_tables
        statement = "SELECT * FROM paragraph_references WHERE reference_type = 'table' AND paper_arxiv_id = %s;"
        cur.execute(statement, (arxiv_id,))
        
        para_tables = cur.fetchall()
        
        if not para_tables:
            print(f"No table references found for arxiv_id: {arxiv_id}")
            continue
        
        # Find two things: One is the exact paragraph, the other is the exact table
        for para_table in para_tables:
            # Assuming the structure of paragraph_references table
            # Adjust indices based on your actual table structure
            paragraph_id = para_table[1]
            paper_section = para_table[2]
            reference_label = para_table[4]  # Assuming this is the actual table reference
            reference_type = 'table'  # Since we're filtering by reference_label = 'table'
            
            label = f"\\label{{{reference_label}}}"  # Fixed: escaped backslash
            
            # First search for the global paragraph id
            statement2 = "SELECT id FROM paragraphs WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s;"
            cur.execute(statement2, (arxiv_id, paper_section, paragraph_id))
            
            global_paragraph_result = cur.fetchone()
            global_paragraph_id = global_paragraph_result[0] if global_paragraph_result else None
            
            # Then we fetch table id (changed from figures to tables)
            statement3 = "SELECT id FROM tables WHERE label = %s AND paper_arxiv_id = %s;"
            cur.execute(statement3, (label, arxiv_id))
            
            table_result = cur.fetchone()
            table_id = table_result[0] if table_result else None
            
            # Save this line to csv file
            row = [
                row_id,
                paragraph_id,
                paper_section,
                arxiv_id,
                reference_label,
                reference_type,
                global_paragraph_id,
                table_id  # Changed from figure_id to table_id
            ]
            csv_writer.writerow(row)
            row_id += 1
        
        print(f"Processed {len(para_tables)} table references for arxiv_id: {arxiv_id}")

# Close database connection
cur.close()
conn.close()

print(f"Data extraction completed. Results saved to {dest_csv_path}")