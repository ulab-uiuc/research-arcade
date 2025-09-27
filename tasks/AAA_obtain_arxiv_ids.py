import json
import psycopg2

save_path = "./json/arxiv_id_insertion.json"

statement = """SELECT p.arxiv_id
FROM papers p
JOIN paragraph_references pr ON pr.paper_arxiv_id = p.arxiv_id
JOIN (SELECT paper_arxiv_id, COUNT(*) as fig_count 
      FROM figures GROUP BY paper_arxiv_id HAVING COUNT(*) > 9) f 
      ON f.paper_arxiv_id = p.arxiv_id
JOIN (SELECT paper_arxiv_id, COUNT(*) as tab_count 
      FROM tables GROUP BY paper_arxiv_id HAVING COUNT(*) > 4) t 
      ON t.paper_arxiv_id = p.arxiv_id
WHERE pr.reference_type IN ('figure', 'table')
GROUP BY p.arxiv_id
HAVING SUM(CASE WHEN pr.reference_type = 'figure' THEN 1 ELSE 0 END) > 9
   AND SUM(CASE WHEN pr.reference_type = 'table' THEN 1 ELSE 0 END) > 4;
"""

conn = psycopg2.connect(
    host="localhost",
    port="5433",
    dbname="postgres",
    user="cl195"
)
# Enable autocommit
conn.autocommit = True
cur = conn.cursor()

cur.execute(statement)
result = cur.fetchall()

# Extract just the arxiv_id strings from the tuples
arxiv_ids = [row[0] for row in result]

# Fixed: Use json.dump with correct parameter order
with open(save_path, 'w') as f:
    json.dump(arxiv_ids, f, indent=2)

# Close connections
cur.close()
conn.close()

print(f"Saved {len(arxiv_ids)} arxiv IDs to {save_path}")


