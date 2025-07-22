"""
Serializer demo
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphserializer.database_serializer import DatabaseSerializer

ds = DatabaseSerializer()

sql = """
SELECT * FROM authors
"""

path1 = "data_json/authors.json"
path2 = "data_json/authors.csv"
path3 = "data_json/authors.xml"

ds.query_to_json_file(query=sql, output_path=path1)
ds.query_to_csv_file(query=sql, output_path=path2)
ds.query_to_xml_file(query=sql, output_path=path3)

