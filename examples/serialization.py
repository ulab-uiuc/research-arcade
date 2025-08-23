import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphserializer.database_serializer import DatabaseSerializer

ds = DatabaseSerializer()

path = "./csv"
table_name = "paragraphs_2508.00450"
file_name = f"{table_name}_csv.csv"

# output_path = ds.all_to_csv(table_name=table_name, file_name=file_name, dir_path=path, max_results=60000)
# print(output_path)

output_path = f"{path}/{file_name}"
# query = """
# SELECT p.*
# FROM paragraphs AS p
# WHERE char_length(p.content) > 200
#   AND (
#     EXISTS (
#       SELECT 1 FROM paragraph_references pr
#       WHERE pr.paragraph_id = p.paragraph_id AND pr.paper_arxiv_id = p.paper_arxiv_id AND pr.reference_type = 'figure'
#     )
#     AND
#     EXISTS (
#       SELECT 1 FROM paragraph_references pr
#       WHERE pr.paragraph_id = p.paragraph_id AND pr.paper_arxiv_id = p.paper_arxiv_id AND pr.reference_type = 'table'
#     )
#   );
# """

query = """
SELECT * FROM paragraphs WHERE paper_arxiv_id = '2508.00450'
"""

ds.query_to_csv_file(query=query, output_path=output_path)

def main(args):
    pass