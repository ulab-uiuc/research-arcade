import json
import psycopg2
from psycopg2.extras import Json
from typing import Callable, Optional

class Database:
    # … all your existing connection and table-creation methods …

    def __init__(self):
        
        
        pass

    def serialize_table(
        self,
        table_name: str,
        schema: str = 'public',
        parser: Optional[Callable[[str], str]] = None
    ) -> list[dict]:
        """
        Fetch all rows from `schema.table_name` and return a list of dicts.
        If `parser` is provided, apply it to each string field.
        """
        sql = f'SELECT * FROM "{schema}"."{table_name}"'
        with self.conn.cursor() as cur:
            cur.execute(sql)
            return self._dict_from_cursor(cur, parser)

    def serialize_all(
        self,
        tables: Optional[list[str]] = None,
        schema: str = 'public',
        parser: Optional[Callable[[str], str]] = None
    ) -> dict:
        """
        Dump multiple tables into a single dict, applying `parser` to string fields.
        - tables: list of table names; if None, will use default list.
        Returns { table_name: [ {col: val,…}, … ], … }
        """
        if tables is None:
            tables = [
                'papers', 'sections', 'authors', 'categories', 'institutions',
                'figures', 'tables', 'paper_authors', 'paper_category',
                'citations', 'paper_figures', 'paper_tables', 'author_affiliation'
            ]
        export_data = {}
        for t in tables:
            export_data[t] = self.serialize_table(t, schema, parser)
        return export_data

    def export_to_json(
        self,
        path: str,
        tables: Optional[list[str]] = None,
        parser: Optional[Callable[[str], str]] = None,
        **json_kwargs
    ) -> None:
        """
        Write out your entire schema (or a subset) to JSON file.
        Applies `parser` to string fields if provided.
        """
        data = self.serialize_all(tables, parser=parser)
        with open(path, 'w') as f:
            json.dump(data, f, **json_kwargs)

# Example usage:
if __name__ == '__main__':
    def latex_parser(raw: str) -> str:
        # e.g. integrate your LaTeX-to-text parser here
        from pylatexenc.latex2text import LatexNodes2Text
        return LatexNodes2Text().latex_to_text(raw)

    db = Database()
    # Serialize 'sections' table, parsing LaTeX in content/title columns
    sections = db.serialize_table('sections', parser=latex_parser)
    print(f"Loaded {len(sections)} sections (LaTeX parsed).")

    # Full export with parsing
    db.export_to_json('db_dump_parsed.json', parser=latex_parser, indent=2)
    print("Exported database with parsed fields.")
    db.close()
