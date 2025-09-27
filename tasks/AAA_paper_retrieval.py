import json
import psycopg2
import argparse
from collections import defaultdict

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

# Load arxiv IDs
with open(arxiv_path, 'r') as f:
    arxiv_ids = list(json.load(f))

print(f"Processing {len(arxiv_ids)} papers...")

# Single query to get all paper, section, and paragraph data
main_query = """
SELECT 
    p.arxiv_id,
    p.title as paper_title,
    s.title as section_title,
    par.paragraph_id,
    par.content
FROM papers p
LEFT JOIN sections s ON p.arxiv_id = s.paper_arxiv_id
LEFT JOIN paragraphs par ON p.arxiv_id = par.paper_arxiv_id AND s.title = par.paper_section
WHERE p.arxiv_id = ANY(%s)
ORDER BY p.arxiv_id, s.title, par.paragraph_id;
"""

# Query to get all figure references
figure_refs_query = """
SELECT 
    pr.paper_arxiv_id,
    pr.paper_section,
    pr.paragraph_id,
    pr.reference_label,
    f.caption,
    f.path
FROM paragraph_references pr
LEFT JOIN figures f ON CONCAT('\label{', pr.reference_label, '}') = f.label AND pr.paper_arxiv_id = f.paper_arxiv_id
WHERE pr.paper_arxiv_id = ANY(%s) 
    AND pr.reference_type = 'figure'
ORDER BY pr.paper_arxiv_id, pr.paper_section, pr.paragraph_id;
"""

# Query to get all table references
table_refs_query = """
SELECT 
    pr.paper_arxiv_id,
    pr.paper_section,
    pr.paragraph_id,
    pr.reference_label,
    t.caption,
    t.table_text
FROM paragraph_references pr
LEFT JOIN tables t ON CONCAT('\label{', pr.reference_label, '}') = t.label AND pr.paper_arxiv_id = t.paper_arxiv_id
WHERE pr.paper_arxiv_id = ANY(%s) 
    AND pr.reference_type = 'table'
ORDER BY pr.paper_arxiv_id, pr.paper_section, pr.paragraph_id;
"""

# Execute all queries
print("Fetching main paper data...")
cur.execute(main_query, (arxiv_ids,))
main_results = cur.fetchall()

print("Fetching figure references...")
cur.execute(figure_refs_query, (arxiv_ids,))
figure_results = cur.fetchall()
print(f"Retrieved {len(figure_results)} figure reference rows")

print("Fetching table references...")
cur.execute(table_refs_query, (arxiv_ids,))
table_results = cur.fetchall()
print(f"Retrieved {len(table_results)} table reference rows")

# Debug: Show first few results
if figure_results:
    print("Sample figure results:")
    for i, row in enumerate(figure_results[:3]):
        arxiv_id, section, para_id, ref_label, caption, path = row
        print(f"  {i+1}: arxiv_id={arxiv_id}, ref_label='{ref_label}', caption={'Present' if caption else 'NULL'}, path={'Present' if path else 'NULL'}")

if table_results:
    print("Sample table results:")
    for i, row in enumerate(table_results[:3]):
        arxiv_id, section, para_id, ref_label, caption, table_text = row
        print(f"  {i+1}: arxiv_id={arxiv_id}, ref_label='{ref_label}', caption={'Present' if caption else 'NULL'}, table_text={'Present' if table_text else 'NULL'}")
        
if not figure_results and not table_results:
    print("No figure or table results found - this suggests the JOINs are not matching any records")

# Close database connection
cur.close()
conn.close()

print("Processing results...")

# Create lookup dictionaries for figures and tables
figure_lookup = defaultdict(list)
for arxiv_id, section, para_id, ref_label, caption, path in figure_results:
    if ref_label:  # Only add if reference exists
        figure_lookup[(arxiv_id, section, para_id)].append({
            "label": f"\\label{{{ref_label}}}",
            "caption": caption,
            "path": path
        })

table_lookup = defaultdict(list)
for arxiv_id, section, para_id, ref_label, caption, table_text in table_results:
    if ref_label:  # Only add if reference exists
        table_lookup[(arxiv_id, section, para_id)].append({
            "label": f"\\label{{{ref_label}}}",
            "caption": caption,
            "table_text": table_text
        })

# Build hierarchical structure
papers_dict = defaultdict(lambda: {"sections": defaultdict(lambda: {"paragraphs": {}})})

# Process main results
for arxiv_id, paper_title, section_title, para_id, content in main_results:
    # Set paper title
    if "title" not in papers_dict[arxiv_id]:
        papers_dict[arxiv_id]["title"] = paper_title
        papers_dict[arxiv_id]["arxiv_id"] = arxiv_id
    
    # Set section title
    if section_title and "section_title" not in papers_dict[arxiv_id]["sections"][section_title]:
        papers_dict[arxiv_id]["sections"][section_title]["section_title"] = section_title
    
    # Add paragraph if it exists
    if para_id is not None and content is not None:
        # Get figures and tables for this paragraph
        figures = figure_lookup.get((arxiv_id, section_title, para_id), [])
        tables = table_lookup.get((arxiv_id, section_title, para_id), [])
        
        papers_dict[arxiv_id]["sections"][section_title]["paragraphs"][para_id] = {
            "paragraph_id": para_id,
            "content": content,
            "figures": figures,
            "tables": tables
        }

# Convert to final format
papers = []
for arxiv_id in arxiv_ids:  # Maintain original order
    if arxiv_id in papers_dict:
        paper_data = papers_dict[arxiv_id]
        
        # Convert sections dict to list
        sections = []
        for section_title, section_data in paper_data["sections"].items():
            if section_title:  # Skip None sections
                # Convert paragraphs dict to list, sorted by paragraph_id
                paragraphs = []
                for para_id in sorted(section_data["paragraphs"].keys()):
                    paragraphs.append(section_data["paragraphs"][para_id])
                
                sections.append({
                    "section_title": section_title,
                    "paragraphs": paragraphs
                })
        
        papers.append({
            "title": paper_data["title"],
            "arxiv_id": paper_data["arxiv_id"],
            "sections": sections
        })

# Save to JSON file
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)

print(f"Data successfully saved to {output_path}")
print(f"Processed {len(papers)} papers")