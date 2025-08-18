import psycopg2
from psycopg2 import sql


from openai import OpenAI
import os
import os
import re

PASSWORD = "Lcs20031121!"


def select_papers_with_criteria(
    min_paragraph=10, min_figure=0, min_citation=0, min_table=0
):
    """
    Return arxiv_ids from papers that meet all provided minimum counts.
    Assumes child tables link to papers via columns named *_arxiv_id.
    Adjust column/table names if yours differ.
    """
    conn = psycopg2.connect(
            host="localhost", dbname="postgres",
            user="postgres", password=PASSWORD, port="5432"
    )
    try:
        conditions = []
        params = []

        if min_paragraph > 0:
            conditions.append("""
                (SELECT COUNT(*)
                FROM paragraphs pr
                WHERE pr.paper_arxiv_id = p.arxiv_id) >= %s
            """)
            params.append(min_paragraph)

        if min_figure > 0:
            conditions.append("""
                (SELECT COUNT(*)
                FROM figures fg
                WHERE fg.paper_arxiv_id = p.arxiv_id) >= %s
            """)
            params.append(min_figure)

        if min_citation > 0:
            conditions.append("""
                (SELECT COUNT(*)
                FROM citations ct
                WHERE ct.citing_arxiv_id = p.arxiv_id) >= %s
            """)
            params.append(min_citation)

        if min_table > 0:
            conditions.append("""
                (SELECT COUNT(*)
                FROM tables tb
                WHERE tb.paper_arxiv_id = p.arxiv_id) >= %s
            """)
            params.append(min_table)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT p.arxiv_id
              FROM papers AS p
              {where_clause}
        """

        with conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

def load_client(api_key = None, base_url = "https://integrate.api.nvidia.com/v1"):

    if api_key is None:
        api_key = os.environ.get("API_KEY")
        if api_key is None:
            raise ValueError("API key must be provided or set in environment variable API_KEY.")
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    return client


def answer_evaluation(answer, ground_truth, data_type=None):

    indexes = answer.split(',')
    index_number = []
    for index in indexes:
        index = index.strip()
        if not index:
            continue
        try:
            index_number.append(int(index))
        except Exception as e:
            print(f"Invalid index '{index}': {e}")

    predicted = set(index_number)
    truth = set(ground_truth)

    tp = len(predicted & truth)  # true positives
    fp = len(predicted - truth)  # false positives
    fn = len(truth - predicted)  # false negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted": predicted,
        "ground_truth": truth
    }


def paragraph_ref_to_global_ref(paragraph_id, ref_type):
    """
    Given a paragraph_id and ref_type ('figure' or 'table'), return a mapping
    from (paper_arxiv_id, reference_label) to the global ID in the corresponding
    table ('figures' or 'tables').
    """
    if ref_type not in {"figure", "table"}:
        raise ValueError(f"Unsupported reference type: {ref_type}")

    table_map = {"figure": "figures", "table": "tables"}
    table_name = table_map[ref_type]

    conn = psycopg2.connect(
        host="localhost",
        dbname="postgres",
        user="postgres",
        password=PASSWORD,
        port="5432"
    )

    ref_id_mapping = {}

    try:
        with conn:
            with conn.cursor() as cur:
                # 1) Get (paper_arxiv_id, reference_label) pairs
                cur.execute(
                    """
                    SELECT paper_arxiv_id, reference_label
                    FROM paragraph_references
                    WHERE paragraph_id = %s AND reference_type = %s
                    """,
                    (paragraph_id, ref_type)
                )
                pairs = cur.fetchall()

                if not pairs:
                    return ref_id_mapping  # empty dict

                # 2) Resolve each reference
                for arxiv_id, ref_label in pairs:
                    formatted_label = figure_label_add_latex_format(ref_label)

                    cur.execute(
                        sql.SQL("""
                            SELECT id FROM {}
                            WHERE paper_arxiv_id = %s AND label = %s
                        """).format(sql.Identifier(table_name)),
                        (arxiv_id, formatted_label)
                    )
                    row = cur.fetchone()
                    if row is not None:
                        # map (arxiv_id, original ref_label) -> global_id
                        ref_id_mapping[(arxiv_id, ref_label)] = row[0]

        return ref_id_mapping

    finally:
        conn.close()


def paragraph_ref_id_to_global_ref(paragraph_ref_id, ref_type):
    """
    Given a specific paragraph_references.id and ref_type ('figure' or 'table'),
    return the corresponding global id from figures/tables.
    Returns: int | None
    """
    if ref_type not in {"figure", "table"}:
        raise ValueError(f"Unsupported reference type: {ref_type}")

    table_name = {"figure": "figures", "table": "tables"}[ref_type]

    conn = psycopg2.connect(
        host="localhost",
        dbname="postgres",
        user="postgres",
        password=PASSWORD,
        port="5432"
    )

    try:
        with conn, conn.cursor() as cur:
            # 1) Fetch the (paper_arxiv_id, reference_label) for this specific paragraph_references row
            cur.execute(
                """
                SELECT paper_arxiv_id, reference_label
                FROM paragraph_references
                WHERE id = %s AND reference_type = %s
                """,
                (paragraph_ref_id, ref_type)
            )
            row = cur.fetchone()
            if not row:
                return None

            arxiv_id, ref_label = row

            # Normalize label if your figures/tables.label uses \label{...}
            formatted_label = figure_label_add_latex_format(ref_label)

            # 2) Look up the global id in figures/tables
            cur.execute(
                sql.SQL("""
                    SELECT id FROM {}
                    WHERE paper_arxiv_id = %s AND label = %s
                """).format(sql.Identifier(table_name)),
                (arxiv_id, formatted_label)
            )
            hit = cur.fetchone()
            return hit[0] if hit else None
    finally:
        conn.close()


def paragraph_citation_to_global_citation(paragraph_id):

    conn = psycopg2.connect(
        host="localhost",
        dbname="postgres",
        user="postgres",
        password=PASSWORD,
        port="5432"
    )

    abstract_list = []
    
    try:
        with conn:
            with conn.cursor() as cur:
                # 1) Get (paper_arxiv_id, bib_key) pairs
                cur.execute(
                    """
                    SELECT paper_arxiv_id, bib_key
                    FROM paragraph_citations
                    WHERE paragraph_id = %s
                    """,
                    (paragraph_id,)
                )
                pairs = cur.fetchall()

                if not pairs:
                    return ref_id_mapping  # empty dict

                # 2) Resolve each reference
                for arxiv_id, bib_key in pairs:

                    cur.execute(
                        sql.SQL("""
                            SELECT cited_arxiv_id, bib_title FROM citations
                            WHERE paper_arxiv_id = %s AND bib_key = %s
                        """),
                        (arxiv_id, bib_key)
                    )
                    row = cur.fetchone()
                    if row is not None:

        return ref_id_mapping

    finally:
        conn.close()


def locate_reference(reference_id):
    try:
        with conn, conn.cursor() as cur:
            # 1) Fetch the (paper_arxiv_id, reference_label) for this specific paragraph_refenrences row
            cur.execute(
                """
                SELECT paragraph_id
                FROM paragraph_references
                WHERE id = %s AND reference_type = %s
                """,
                (paragraph_ref_id, ref_type)
            )
        


def figure_latex_path_to_path(path, arxiv_id, latex_path):
    # 2. Replace any forward or backward slash with underscore
    latex_path = re.sub(r'[/]', '_', latex_path)
    # 3. Build the filename with the arXiv ID prefix
    return f"{path}/output/figures/{arxiv_id}/{latex_path}"

def figure_label_add_latex_format(name):
    return f"\\label{{{name}}}"

def figure_label_remove_latex_format(name):
    match = re.match(r"\\label\{(.+?)\}", name.strip())
    if match:
        return match.group(1)
    return None


def load_prompt(data_type):
    pass

