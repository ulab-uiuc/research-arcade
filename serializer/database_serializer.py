import json
import csv
from io import StringIO
import psycopg2

PASSWORD = "Lcs20031121!"


class DatabaseSerializer:
    """
    A versatile serializer that converts a list of dict-like database records
    into various formats (JSON, CSV, and XML).
    Usage:
        serializer = DatabaseSerializer(data)
        json_str = serializer.to_json()
        csv_str = serializer.to_csv()
        xml_str = serializer.to_xml(root_element="records", item_element="record")
    """

    def __init__(self):
        """
        Initialize with a sequence of records, where each record is a dict
        mapping column names to values.
        """

        #First initialize the database connection

        self.conn = psycopg2.connect(
            host="localhost", dbname="postgres",
            user="postgres", password=PASSWORD, port="5432"
        )
        # Enable autocommit
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

    

    def query_to_json_file(self, query, output_path, indent=2):
        """
        Execute a SQL query and write the result as a JSON file.

        :param connection: A DB-API compatible database connection object.
        :param query: SQL query string to execute.
        :param output_path: Path to the output JSON file.
        :param indent: Number of spaces for pretty-printing JSON. None for compact.
        :return: The path to the written JSON file.
        """
        
        self.cur.execute(query)

        # Fetch column names from cursor description
        columns = [desc[0] for desc in self.cur.description]

        # Build list of dict records
        records = [dict(zip(columns, row)) for row in self.cur.fetchall()]

        # Serialize to JSON
        json_str = json.dumps(records, indent=indent)

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

        return output_path
