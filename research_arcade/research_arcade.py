import os
import pandas as pd
from .sql_database.arxiv_sql_database import ArxivSQLDatabase

class ResearchArcade:
    def __init__(self, has_csv_dataset, csv_dir_path, has_sql_dataset, sql_args):
        if has_csv_dataset:
            self.csv_dir_path = csv_dir_path
        
        if has_sql_dataset:
            host, port, dbname, user, password, autocommit = (
                sql_args.host,
                sql_args.port,
                sql_args.dbname,
                sql_args.user,
                sql_args.password,
                sql_args.autocommit,
            )
            try:
                self.arxiv_sql_database = ArxivSQLDatabase(
                    host=host,
                    port=port,
                    dbname=dbname,
                    user=user,
                    password=password,
                    autocommit=autocommit,
                )
            except Exception as e:
                print(f"SQL dataset setup failed: {e}")
                self.arxiv_sql_database = None
        else:
            self.arxiv_sql_database = None

    def sql_dataset_to_csv_dataset(self, schema):
        if self.arxiv_sql_database is None:
            print("SQL database is not initialized.")
            return

        try:
            conn = self.arxiv_sql_database._get_connection()
            conn.autocommit = True
            cur = conn.cursor()

            # Get all table names in the public schema
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s;
            """, (schema,))
            tables = [row[0] for row in cur.fetchall()]

            if not tables:
                print("No tables found in PostgreSQL database.")
                return

            # Ensure CSV directory exists
            os.makedirs(self.csv_dir_path, exist_ok=True)

            for table_name in tables:
                query = f"SELECT * FROM {table_name};"
                df = pd.read_sql_query(query, conn)
                csv_path = os.path.join(self.csv_dir_path, f"{table_name}.csv")
                df.to_csv(csv_path, index=False)
                print(f"Exported table '{table_name}' to '{csv_path}'")

            cur.close()
            conn.close()

        except Exception as e:
            print(f"Error exporting SQL tables to CSV: {e}")
