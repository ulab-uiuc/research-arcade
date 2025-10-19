import sys
import os
import json
import argparse
from types import SimpleNamespace  

from dotenv import load_dotenv  
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research_arcade.research_arcade import ResearchArcade


def main():
    env = {
        "HOST": os.getenv("HOST"),
        "PORT": int(os.getenv("PORT")),
        "DBNAME": os.getenv("DBNAME"),
        "USER": os.getenv("USER"),
        "PASSWORD": os.getenv("PASSWORD"),
        "ARXIV_SCHEMA": os.getenv("ARXIV_SCHEMA", "public"),
    }
    sql_args = SimpleNamespace(
        host=env["HOST"],
        port=env["PORT"],
        dbname=env["DBNAME"],
        user=env["USER"],
        password=env["PASSWORD"],   
        autocommit=True            
    )

    ra = ResearchArcade(
        has_csv_dataset=True,
        csv_dir_path="./arxiv_csv",
        has_sql_dataset=True,
        sql_args=sql_args
    )

    ra.sql_dataset_to_csv_dataset(schema=env["ARXIV_SCHEMA"])


if __name__ == "__main__":
    main()
