import sys, os, gzip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphserializer.database_serializer import DatabaseSerializer

ds = DatabaseSerializer()

out = "./csv/paragraphs_with_figures_tables.csv.gz"
os.makedirs(os.path.dirname(out), exist_ok=True)

statement = """
SELECT p.*
FROM paragraphs AS p
WHERE char_length(p.content) > 200
  AND EXISTS (
    SELECT 1 FROM paragraph_references pr
    WHERE pr.paragraph_id = p.paragraph_id
      AND pr.paper_arxiv_id = p.paper_arxiv_id
      AND pr.paper_section  = p.paper_section
      AND pr.reference_type = 'figure'
  )
  AND EXISTS (
    SELECT 1 FROM paragraph_references pr
    WHERE pr.paragraph_id = p.paragraph_id
      AND pr.paper_arxiv_id = p.paper_arxiv_id
      AND pr.paper_section  = p.paper_section
      AND pr.reference_type = 'table'
  )
""".strip()

copy_sql = f"COPY ({statement}) TO STDOUT WITH CSV HEADER"

with ds.conn.cursor() as cur:
    # Per-statement tuning (session-local, no special privileges)
    cur.execute("""
        SET LOCAL work_mem = '1GB';
        SET LOCAL enable_hashjoin = off;
        SET LOCAL enable_mergejoin = off;
        SET LOCAL enable_sort = off;
    """)
    with gzip.open(out, "wt", newline="") as f:
        cur.copy_expert(copy_sql, f)

print("Wrote:", out)
