import psycopg2

class SQLArxivPaperTable:
    def __init__(self, sql_args):
        """
        sql_args should provide: host, port, dbname, user, password, autocommit (bool)
        """
        self.host, self.port, self.dbname, self.user, self.password, self.autocommit = (
            sql_args.host,
            sql_args.port,
            sql_args.dbname,
            sql_args.user,
            sql_args.password,
            sql_args.autocommit,
        )

    def _get_connection(self):
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )
        conn.autocommit = self.autocommit
        return conn

    def create_paper_tables_table(self):
        """
        Create the arxiv_paper_tables table with a composite uniqueness constraint.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paper_tables (
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    table_id INTEGER NOT NULL
                )
            """)
            # Composite unique index mirrors CSV conflict check
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_arxiv_paper_tables_unique
                ON arxiv_paper_tables (paper_arxiv_id, table_id)
            """)
            cur.close()
        finally:
            conn.close()

    def insert_paper_table(self, paper_arxiv_id, table_id):
        """
        Insert a (paper_arxiv_id, table_id) edge.
        Returns True if inserted, False if the pair already exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_paper_tables (paper_arxiv_id, table_id)
                VALUES (%s, %s)
                ON CONFLICT ON CONSTRAINT ux_arxiv_paper_tables_unique DO NOTHING
                """,
                (paper_arxiv_id, table_id)
            )
            inserted = cur.rowcount > 0  # rowcount==1 if inserted, 0 if conflict
            cur.close()
            return inserted
        finally:
            conn.close()
