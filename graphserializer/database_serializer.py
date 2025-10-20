import json
import csv
import psycopg2



class DatabaseSerializer:
    """
    A versatile serializer that connects to a PostgreSQL database,
    runs SQL queries, and exports results in JSON, CSV, or XML formats.

    Usage examples:
        ds = DatabaseSerializer()
        ds.query_to_json_file("SELECT * FROM authors;", "data_json/authors.json")
        ds.query_to_csv_file("SELECT * FROM books;", "data_csv/books.csv")
        ds.query_to_xml_file("SELECT * FROM users;", "data_xml/users.xml")
    """

    def __init__(self):
        # Initialize the database connection
        self.conn = psycopg2.connect(
            host="localhost",
            port="5433",
            dbname="postgres",
            user="cl195"
        )
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

    def _fetch_records(self, query, parameters=None):
        """
        Helper to execute a query and return a list of dict records.
        """
        self.cur.execute(query, parameters)
        columns = [desc[0] for desc in self.cur.description]
        return [dict(zip(columns, row)) for row in self.cur.fetchall()]
        
    def query_to_json_file(self, query, output_path, parameters=None, indent=2):
        """
        Execute a SQL query and write the result as a JSON file.
        
        :param query: SQL query string to execute.
        :param output_path: Path to write the JSON output.
        :param indent: Pretty-print indentation (None for compact).
        :return: The output path.
        """
        records = self._fetch_records(query, parameters)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=indent)
        return output_path
        
    def query_to_csv_file(self, query, output_path, parameters=None, delimiter=',', include_header=True):
        """
        Execute a SQL query and write the result as a CSV file.

        :param query: SQL query string to execute.
        :param output_path: Path to write the CSV output.
        :param delimiter: Field delimiter (default comma).
        :param include_header: Whether to write column headers.
        :return: The output path.
        """
        records = self._fetch_records(query, parameters)
        # if not records:
        #     # create empty file
        #     open(output_path, 'w').close()
        #     return output_path

        fieldnames = list(records[0].keys())
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            if include_header:
                writer.writeheader()
            for row in records:
                writer.writerow(row)
        return output_path

    def query_to_xml_file(self, query, output_path, root_element='data', item_element='item'):
        """
        Execute a SQL query and write the result as an XML file.

        :param query: SQL query string to execute.
        :param output_path: Path to write the XML output.
        :param root_element: XML root tag name.
        :param item_element: XML tag for each record.
        :return: The output path.
        """
        def _escape(s):
            return (str(s)
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&apos;'))

        records = self._fetch_records(query)
        xml_lines = [f'<{root_element}>']
        for record in records:
            xml_lines.append(f'  <{item_element}>')
            for key, value in record.items():
                xml_lines.append(f'    <{key}>{_escape(value)}</{key}>')
            xml_lines.append(f'  </{item_element}>')
        xml_lines.append(f'</{root_element}>')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(xml_lines))
        return output_path
    
    def all_to_csv(self, table_name, file_name, dir_path, max_results=100):

        sql = f"""
        SELECT * FROM {table_name}
        LIMIT {max_results}
        """
        \
        dest_path = f"{dir_path}/{file_name}"

        op = self.query_to_csv_file(query=sql, output_path=dest_path)

        return op
