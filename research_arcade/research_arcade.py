
"""
Main interface
"""

"""
Configuration, including:
1. CSV setting:
    1.1 CSV path
2. PostgreSQL setting:
    2.0 if we use it
    2.1 host
    2.2 port
    2.3 dbname
    2.4 user
    2.5 password
    2.6 autocommit
3. Other configuration
    3.1 keys(Maybe? or we can use env)
"""

"""
So... there should be three modes

1. CSV only
2. PostgreSQL only
3. CSV+PostgreSQL, where CSV takes data first, and gives them to PostgreSQL. CSV serves as a data pool for PostgreSQL
"""

"""
Interaction between CSV and PostgreSQL
1. Update CSV data into PostgreSQL
2. ???
"""
from csv_database.arxiv_papers import ArxivCSVDataset
from sql_database.arxiv_papers import ArxivSQLDatabase


class ResearchArcade:

    def __init__(self, has_csv_dataset, csv_path, has_sql_dataset, sql_args):
        # Setup the csv dataset class

        if has_csv_dataset:
            self.csv_path = csv_path
            self.arxiv_csv_dataset = ArxivCSVDataset(self.csv_path)

        if has_sql_dataset:
            # Unzip the sql_args
            """
            sql_args:
            host, port, dbname, 
                 user, password, autocommit
            """
            host, port, dbname, user, password, autocommit = sql_args.host, sql_args.port, sql_args.dbname, sql_args.user, sql_args.password, sql_args.autocommit
            
            try:
                self.arxiv_sql_database = ArxivSQLDatabase(host=host, port=port, dbname=dbname, user=user, password=password, autocommit=autocommit)
            except Exception:
                print("SQL dataset setup failed.")
                self.arxiv_sql_database = None
        else:
            self.arxiv_sql_database = None

        