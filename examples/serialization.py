import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphserializer.database_serializer import DatabaseSerializer

ds = DatabaseSerializer()

path = "./csv"
table_name = "arxiv_ids_ct"
file_name = f"{table_name}_csv.csv"

# output_path = ds.all_to_csv(table_name=table_name, file_name=file_name, dir_path=path, max_results=60000)
# print(output_path)

output_path = f"{path}/{file_name}"
query = """SELECT arxiv_id FROM papers;"""

ds.query_to_csv_file(query=query, output_path=output_path)

def main(args):
    pass