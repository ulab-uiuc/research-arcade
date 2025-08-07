import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper_crawler.crawler_job import CrawlerJob
from graphserializer.database_serializer import DatabaseSerializer

ds = DatabaseSerializer()

path = "download/csv"
table_name = "paragraph_references"
file_name = f"{table_name}_csv.csv"

output_path = ds.all_to_csv(table_name=table_name, file_name=file_name, dir_path=path, max_results=1000)
print(output_path)
