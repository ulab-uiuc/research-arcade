from load_model import load_model


def missing_data_imputation(args):

    """
    1. Model name
    2. Data type: figure, table, citation
    3. Data id (fig/tb/cit/ect.) which cit information are we trying to predict. What are we going to remove
    4. Link: which link (another kind of link) are we going to remove
    5. 
    """

    conn = psycopg2.connect(
            host="localhost", dbname="postgres",
            user="postgres", password=PASSWORD, port="5432"
    )
    cur = conn.cursor()

    model_name = args.model_name

    model = load_model(model_name)

    data_type = args.data_type

    table_name = None
    id_name = "id" # The primary key

    # if data_type == "figure":
    #     table_name = "figures"
    # elif data_type == "table":
    #     table_name = "tables"
    # elif data_type == "citations":
    #     table_name = "citations"

    data_id_path = args.data_id_path

    # Load data_ids

    data_ids = load_ids(data_id_path)


    if data_type == "figure":
        for data_id in data_ids:
            # Figure path
            figure_path, arxiv_id = cur.execute("""
            SELECT file_path, paper_arxiv_id FROM figures WHERE id = %s
            """, (data_id,))

            # Two ways: if the model is a visual-language model, give the photo to the model directly; if the model is a language only model, another visual model should be provided for caption generation


    elif data_type == "table":
        for data_id in data_ids:
            # select the context
            table_context = cur.execute("""
            SELECT table_text, paper_arxiv_id FROM tables WHERE id = %s
            """, (data_id,))

            # First give the model the table context, then provide the model with each paragraphs.
    
    elif data_type == "citations":
        for data_id in data_ids:
            cited_arxiv_id = cur.execute("""
            SELECT cited_arxiv_id, paper_arxiv_id FROM tables WHERE id = %s
            """, (data_id,))

            # Remove the curl braces and signs from the bib_title
            # Assume that the paper with specified arxiv id exists in the database, fetch it

            cited_paper_title, cited_paper_abstract = cur.execute("""
            SELECT title, abstract FROM papers WHERE arxiv_id = %s
            """, (cited_arxiv_id))


            # First give the model the abstract of cited paper, then provide the model with each paragraphs.
    



    # Query the related information



    # Method: iterative judgement over paragraphs. Go thourough each 



    
    


    