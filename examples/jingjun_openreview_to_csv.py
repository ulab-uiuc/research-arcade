import json
import psycopg2
import os
import sys
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphserializer.database_serializer import DatabaseSerializer
from tasks.utils import paragraph_ref_to_global_ref, paragraph_ref_id_to_global_ref

file_path = "./json/openreview_arxiv_2025.json"


def split_into_n_chunks(seq, n):
    """Split seq into n near-equal chunks (sizes differ by at most 1)."""
    L = len(seq)
    if n <= 0:
        return []
    base, extra = divmod(L, n)
    chunks = []
    start = 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        end = start + size
        chunks.append(seq[start:end])
        start = end
    return chunks


arxiv_ids = []

# Collect all arxiv_ids
for i in range(1, 11):
    file_path = f"./json/arxiv_id_openreview_{i}.json"
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

        # If it's a list of strings
        if data and isinstance(data[0], str):
            arxiv_ids.extend(data)
        # If it's a list of dicts with 'arxiv_id'
        elif data and isinstance(data[0], dict):
            arxiv_ids.extend(d["arxiv_id"] for d in data if "arxiv_id" in d)

print(f"Total arxiv_ids collected: {len(arxiv_ids)}")

# Connect to DB
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    dbname="postgres",
    user="cl195",
)
conn.autocommit = True
cur = conn.cursor()

statement = """
SELECT COUNT(*) 
FROM papers 
WHERE arxiv_id = ANY(%s)
"""

cur.execute(statement, (arxiv_ids,))
count = cur.fetchone()[0]

ds = DatabaseSerializer()

query1 = """
SELECT * FROM papers WHERE arxiv_id = ANY(%s)
"""

file_path1 = "./jingjun/papers.csv"

query2 = """
SELECT * FROM paragraphs WHERE paper_arxiv_id = ANY(%s)
"""

file_path2 = "./jingjun/paragraphs.csv"

query3 = """
SELECT * FROM figures WHERE paper_arxiv_id = ANY(%s)
"""

file_path3 = "./jingjun/figures.csv"

query4 = """
SELECT * FROM tables WHERE paper_arxiv_id = ANY(%s)
"""

file_path4 = "./jingjun/tables.csv"

query5 = """
SELECT * FROM paragraph_references WHERE paper_arxiv_id = ANY(%s) AND reference_type = 'figure'
"""

file_path5 = "./jingjun/paragraph_figure.csv"

query6 = """
SELECT * FROM paragraph_references WHERE paper_arxiv_id = ANY(%s) AND reference_type = 'table'
"""

file_path6 = "./jingjun/paragraph_table.csv"

# Export basic tables


# Enhanced paragraph-figure references with figure details
def enhance_paragraph_figure_references():
    """Enhance paragraph-figure references with figure details."""
    
    # First, get all figure information for quick lookup
    figures_query = """
    SELECT paper_arxiv_id, id, path, caption, label, name 
    FROM figures 
    WHERE paper_arxiv_id = ANY(%s)
    """
    
    cur.execute(figures_query, (arxiv_ids,))
    figures_data = cur.fetchall()
    
    # Create a lookup dictionary: (paper_arxiv_id, figure_id) -> figure_details
    figures_lookup = {}
    for row in figures_data:
        paper_arxiv_id, figure_id, path, caption, label, name = row
        figures_lookup[(paper_arxiv_id, figure_id)] = {
            'path': path,
            'caption': caption,
            'label': label,
            'name': name
        }
    
    print(f"Loaded {len(figures_lookup)} figure records for lookup")
    
    # Read the original paragraph-figure references
    enhanced_file_path = "./jingjun/paragraph_figure_enhanced.csv"
    
    with open(file_path5, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        original_fieldnames = reader.fieldnames
        []
        # Add new fields
        enhanced_fieldnames = list(original_fieldnames) + ['figure_path', 'figure_caption', 'figure_label', 'figure_name']
        
        with open(enhanced_file_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=enhanced_fieldnames)
            writer.writeheader()
            
            enhanced_count = 0
            total_count = 0
            
            for row in reader:
                total_count += 1
                paper_arxiv_id = row['paper_arxiv_id']
                # paper_section = row['paper_section']
                # paragraph_id = row['paragraph_id']
                paragraph_ref_id = row['id']
                
                # Get global reference ID
                try:
                    global_ref = paragraph_ref_id_to_global_ref(
                        paragraph_ref_id=paragraph_ref_id, 
                        ref_type='figure'
                    )

                    # Look up figure details
                    figure_key = (paper_arxiv_id, global_ref)

                    if figure_key in figures_lookup:
                        figure_details = figures_lookup[figure_key]
                        row['figure_path'] = figure_details['path']
                        row['figure_caption'] = figure_details['caption']
                        row['figure_label'] = figure_details['label']
                        row['figure_name'] = figure_details['name']
                        enhanced_count += 1
                    else:
                        # Fill with None/empty values if not found
                        row['figure_path'] = None
                        row['figure_caption'] = None
                        row['figure_label'] = None
                        row['figure_name'] = None
                        
                except Exception as e:
                    print(f"Error processing row {total_count}: {e}")
                    # Fill with None values on error
                    row['figure_path'] = None
                    row['figure_caption'] = None
                    row['figure_label'] = None
                    row['figure_name'] = None
                writer.writerow(row)
    
    print(f"Enhanced paragraph-figure references: {enhanced_count}/{total_count} records enhanced")
    print(f"Enhanced file saved to: {enhanced_file_path}")

# Enhanced paragraph-table references with table details
def enhance_paragraph_table_references():
    """Enhance paragraph-table references with table details."""
    
    # First, get all table information for quick lookup
    tables_query = """
    SELECT paper_arxiv_id, id, path, caption, label, table_text 
    FROM tables 
    WHERE paper_arxiv_id = ANY(%s)
    """
    
    cur.execute(tables_query, (arxiv_ids,))
    tables_data = cur.fetchall()
    
    # Create a lookup dictionary: (paper_arxiv_id, table_id) -> table_details
    tables_lookup = {}
    for row in tables_data:
        paper_arxiv_id, table_id, path, caption, label, table_text = row
        tables_lookup[(paper_arxiv_id, table_id)] = {
            'path': path,
            'caption': caption,
            'label': label,
            'table_text': table_text
        }
    
    print(f"Loaded {len(tables_lookup)} table records for lookup")
    
    # Read the original paragraph-table references
    enhanced_file_path = "./jingjun/paragraph_table_enhanced.csv"
    
    with open(file_path6, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        original_fieldnames = reader.fieldnames
        
        # Add new fields
        enhanced_fieldnames = list(original_fieldnames) + ['table_path', 'table_caption', 'table_label', 'table_text']
        
        with open(enhanced_file_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=enhanced_fieldnames)
            writer.writeheader()
            
            enhanced_count = 0
            total_count = 0
            
            for row in reader:
                total_count += 1
                paper_arxiv_id = row['paper_arxiv_id']
                paragraph_ref_id = row['id']
                
                # Get global reference ID
                try:
                    global_ref = paragraph_ref_id_to_global_ref(
                        paragraph_ref_id=paragraph_ref_id, 
                        ref_type='table'
                    )
                    
                    # Look up table details
                    table_key = (paper_arxiv_id, global_ref)
                    if table_key in tables_lookup:
                        table_details = tables_lookup[table_key]
                        row['table_path'] = table_details['path']
                        row['table_caption'] = table_details['caption']
                        row['table_label'] = table_details['label']
                        row['table_text'] = table_details['table_text']
                        enhanced_count += 1
                    else:
                        # Fill with None/empty values if not found
                        row['table_path'] = None
                        row['table_caption'] = None
                        row['table_label'] = None
                        row['table_text'] = None
                        
                except Exception as e:
                    print(f"Error processing row {total_count}: {e}")
                    # Fill with None values on error
                    row['table_path'] = None
                    row['table_caption'] = None
                    row['table_label'] = None
                    row['table_text'] = None
                
                writer.writerow(row)
    
    print(f"Enhanced paragraph-table references: {enhanced_count}/{total_count} records enhanced")
    print(f"Enhanced file saved to: {enhanced_file_path}")

# Run the enhancement functions
print("Enhancing paragraph-figure references...")
enhance_paragraph_figure_references()

print("Enhancing paragraph-table references...")
enhance_paragraph_table_references()

print("All exports completed successfully!")

# Close database connection
cur.close()
conn.close()
