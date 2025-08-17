import psycopg2

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


from openai import OpenAI
import os

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
    

def load_prompt(data_type):
    
