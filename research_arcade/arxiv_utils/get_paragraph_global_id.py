import os
import csv

# Search for the paragraph global id based on 'paragraph_id', 'paper_section_id' and 'paper_arxiv_id' and return it
def get_paragraph_global_id_csv(csv_dir, paper_arxiv_id, paper_section_id, paragraph_id):
    with open(os.path.join(csv_dir, 'arxiv_paragraphs.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['paper_arxiv_id'] == paper_arxiv_id and row['paper_section_id'] == paper_section_id and row['paragraph_id'] == paragraph_id:
                return int(row['paragraph_global_id'])
    return None

def get_paragraph_global_id_sql(conn, paper_arxiv_id, paper_section_id, paragraph_id):
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id FROM arxiv_paragraphs 
            WHERE paper_arxiv_id = %s AND paper_section_id = %s AND paragraph_id = %s
        """, (paper_arxiv_id, paper_section_id, paragraph_id))
        result = cur.fetchone()
        return result[0] if result else None
    finally:
        cur.close()
