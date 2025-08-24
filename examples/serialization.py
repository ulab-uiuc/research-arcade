import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphserializer.database_serializer import DatabaseSerializer

ds = DatabaseSerializer()

path = "./csv"
table_name = "paragraphs_with_figures_tables_citations"
file_name = f"{table_name}_csv.csv"

# output_path = ds.all_to_csv(table_name=table_name, file_name=file_name, dir_path=path, max_results=60000)
# print(output_path)

output_path = f"{path}/{file_name}"

query = """
SELECT p.*
FROM paragraphs AS p
WHERE char_length(p.content) > 200
  AND (
    EXISTS (
      SELECT 1 FROM paragraph_references pr
      WHERE pr.paragraph_id = p.paragraph_id AND pr.paper_arxiv_id = p.paper_arxiv_id AND pr.paper_section = p.paper_section AND pr.reference_type = 'figure'
    )
    AND
    EXISTS (
      SELECT 1 FROM paragraph_references pr
      WHERE pr.paragraph_id = p.paragraph_id AND pr.paper_arxiv_id = p.paper_arxiv_id AND pr.paper_section = p.paper_section AND pr.reference_type = 'table'
    )
    AND
    EXISTS (
      SELECT 1 FROM paragraph_citations pc
      WHERE pc.citing_arxiv_id = p.paragraph_id AND pc.paper_arxiv_id = p.paper_arxiv_id AND pc.paper_section = p.paper_section
    )
  );
"""


query = """
SELECT p.*
FROM paragraphs AS p
WHERE char_length(p.content) > 200
  AND EXISTS (
        SELECT 1
        FROM paragraph_references pr
        WHERE pr.paper_arxiv_id = p.paper_arxiv_id
          AND pr.paper_section   = p.paper_section
          AND pr.paragraph_id    = p.paragraph_id
          AND pr.reference_type  = 'figure'
      )
  AND EXISTS (
        SELECT 1
        FROM paragraph_references pr
        WHERE pr.paper_arxiv_id = p.paper_arxiv_id
          AND pr.paper_section   = p.paper_section
          AND pr.paragraph_id    = p.paragraph_id
          AND pr.reference_type  = 'table'
      )
  AND EXISTS (
        SELECT 1
        FROM paragraph_citations pc
        WHERE pc.citing_arxiv_id = p.paper_arxiv_id   -- was comparing to p.paragraph_id (type mismatch)
          AND pc.paper_section   = p.paper_section
          AND pc.paragraph_id    = p.paragraph_id
      );
"""
# query = """
# SELECT * FROM paragraphs WHERE paper_arxiv_id = '2508.00450'
# """



ds.query_to_csv_file(query=query, output_path=output_path)

def main(args):
    pass