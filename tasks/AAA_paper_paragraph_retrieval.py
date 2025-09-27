import json
import psycopg2
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract hierarchical paper data from database')
parser.add_argument('arxiv_path', help='Path to the JSON file containing arxiv IDs')
parser.add_argument('output_path', help='Path to save the hierarchical papers data JSON file')
args = parser.parse_args()

arxiv_path = args.arxiv_path
output_path = args.output_path

conn = psycopg2.connect(
    host="localhost",
    port="5433",
    dbname="postgres",
    user="cl195"
)

# Enable autocommit
conn.autocommit = True
cur = conn.cursor()

statement0 = """SELECT title FROM papers WHERE arxiv_id = %s"""
statement1 = """SELECT title from sections WHERE paper_arxiv_id = %s;"""
statement2 = """SELECT paragraph_id, content FROM paragraphs WHERE paper_arxiv_id = %s AND paper_section = %s;"""
statement3 = """SELECT reference_label FROM paragraph_references WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s AND reference_type = 'figure';"""
statement4 = """SELECT reference_label FROM paragraph_references WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s AND reference_type = 'table';"""
statement5 = """SELECT caption, path FROM figures WHERE label = %s AND paper_arxiv_id = %s"""
statement6 = """SELECT caption, table_text FROM tables WHERE label = %s AND paper_arxiv_id = %s"""

# To store papers
papers = []

# Load arxiv IDs
with open(arxiv_path, 'r') as f:
    arxiv_ids = list(json.load(f))

for arxiv_id in arxiv_ids:
    print(f"Processing paper: {arxiv_id}")
    
    # Get paper title
    cur.execute(statement0, (arxiv_id,))
    title_result = cur.fetchone()
    paper_title = title_result[0] if title_result else "Unknown Title"
    
    # Get all sections for this paper
    cur.execute(statement1, (arxiv_id,))
    result = cur.fetchall()
    sections = [row[0] for row in result]
    
    # Create paper structure
    paper = {
        "title": paper_title,
        "arxiv_id": arxiv_id,
        "sections": []
    }
    
    for section in sections:
        print(f"  Processing section: {section}")
        
        # Get paragraphs for this section
        cur.execute(statement2, (arxiv_id, section))
        para_pairs = cur.fetchall()  # [(paragraph_id, content), ...]
        
        # Create section structure
        section_data = {
            "section_title": section,
            "paragraphs": []
        }
        
        for paragraph_id, content in para_pairs:
            print(f"    Processing paragraph: {paragraph_id}")
            
            # Get figure references for this paragraph
            cur.execute(statement3, (arxiv_id, section, paragraph_id))
            figure_refs = cur.fetchall()
            figure_labels = [f"\\label{{{row[0]}}}" for row in figure_refs]  # Add \\label{} formatting
            
            # Get table references for this paragraph
            cur.execute(statement4, (arxiv_id, section, paragraph_id))
            table_refs = cur.fetchall()
            table_labels = [f"\\label{{{row[0]}}}" for row in table_refs]  # Add \\label{} formatting
            
            # Get figure data
            figures = []
            for label in figure_labels:
                cur.execute(statement5, (label, arxiv_id))
                figure_data = cur.fetchall()
                for caption, path in figure_data:
                    figures.append({
                        "label": label,
                        "caption": caption,
                        "path": path
                    })
            
            # Get table data
            tables = []
            for label in table_labels:
                cur.execute(statement6, (label, arxiv_id))
                table_data = cur.fetchall()
                for caption, table_text in table_data:
                    tables.append({
                        "label": label,
                        "caption": caption,
                        "table_text": table_text
                    })
            
            # Create paragraph structure
            paragraph_data = {
                "paragraph_id": paragraph_id,
                "content": content,
                "figures": figures,
                "tables": tables
            }
            
            section_data["paragraphs"].append(paragraph_data)
        
        paper["sections"].append(section_data)
    
    papers.append(paper)

# Close database connection
cur.close()
conn.close()

# Save to JSON file
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)

print(f"Data successfully saved to {output_path}")
print(f"Processed {len(papers)} papers")