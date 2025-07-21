"""
Serializer demo
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from serializer.database_serializer import DatabaseSerializer

ds = DatabaseSerializer()

sql = """
SELECT * FROM authors
"""

path = "data_json/authors.json"

ds.query_to_json_file(query=sql, output_path=path)