from util import load_client, answer_evaluation, load_prompt
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"

def ref_insertion(args):

    """
    ref = figure+table+citation
    1. Model name
    2. Data type: figure, table, citation
    3. Data id (fig/tb/cit/ect.) which cit information are we trying to predict. What are we going to remove
    4. Link: which link (another kind of link) are we going to remove
    """

    conn = psycopg2.connect(
            host="localhost", dbname="postgres",
            user="postgres", password=PASSWORD, port="5432"
    )
    cur = conn.cursor()
    
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    
    data_type = args.data_type

    table_name = None
    id_name = "id" # The primary key

    data_ids = args.data_ids


    # Then retrieve the contents of ref

    ref_id_mapping = {}

    for data_id in data_ids
        # First retrieve the id
        ref_id_global = paragraph_ref_id_to_global_ref(data_id)
        # Then find the content
        ref_id_mapping[data_id] = ref_id_global


    ground_truth = []

    # Retrieve the ground truth
    

    evals = []


    if data_type == "figure":
        for data_id in data_ids:    
            paragraph_to_idx = {}

            prompt = load_prompt(data_type)
            ref_id_global = ref_id_mapping[data_id]
            # Figure path
            cur.execute("""
            SELECT file_path, paper_arxiv_id FROM figures WHERE id = %s
            """, (ref_id_global,))

            row = cur.fetchone()
            figure_path, arxiv_id = None, None
            if row:
                figure_path, arxiv_id = row
            

            # Assume that the model is a visual LLM
            image = Image.open(figure_path).convert("RGB")

            prompt = prompt + "Figure:\n"
            prompt = prompt + f"{{{image}}}\n"

            cur.execute("""
            SELECT id, context FROM paragraphs WHERE paper_arxiv_id = %s
            """, (arxiv_id))
            pairs = cur.fetchall()
            
            idx = 1
            prompt = prompt + "Paragraphs:\n"

            for paragraph_id, paragraph_context in pairs:
                # Give the context to model and see if it agrees on insertion (missing data imputation)
                
                paragraph_to_idx[paragraph_id] = idx
                # Append to a string?
                prompt = prompt + f"{idx}. {{{paragraph_context}}}\n"

                idx += 1
            
            prompt = prompt + """"
            ---

            Final Answer (only indexes, comma-separated):
            """

            completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True
            )

            answer = ""
            for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                answer = answer + chunk.choices[0].delta.content.strip()
                # print(chunk.choices[0].delta.content, end="")
            
            # Finally find the ground truth

            cur.execute("""
            SELECT id FROM PARAGRAPHS WHERE 
            (
                SELECT id FROM 
            )
            """)
            
            answer_evaluation = answer_evaluation(model_answer, ground_truth)
            evals.append(answer_evaluation)


    elif data_type == "table":
        for data_id in data_ids:
            prompt = load_prompt(data_type)
            # select the context
            ref_id_global = ref_id_mapping[data_id]
            cur.execute("""
            SELECT table_text, paper_arxiv_id FROM tables WHERE id = %s
            """, (ref_id_global,))

            row = cur.fetchone()

            table_content, arxiv_id = None, None
            if row:
                table_content, arxiv_id = row
            
            prompt = prompt + "Table:\n"
            prompt = prompt + f"{{{table_content}}}\n"
            

            # First give the model the table context, then provide the model with each paragraphs.
            
            # Go through each paragraph and tell if the table should be inserted here
            
            # Return a list of paragraph_id, paragraph_text pairs
            cur.execute("""
            SELECT id, context FROM paragraphs WHERE paper_arxiv_id = %s
            """, (arxiv_id))
            pairs = cur.fetchall()
            
            idx = 1
            prompt = prompt + "Paragraphs:\n"

            for paragraph_id, paragraph_context in pairs:
                # Give the context to model and see if it agrees on insertion (missing data imputation)
                
                paragraph_to_idx[paragraph_id] = idx
                # Append to a string?
                prompt = prompt + f"{idx}. {{{paragraph_context}}}\n"

                idx += 1
            
            prompt = prompt + """"
            ---

            Final Answer (only indexes, comma-separated):
            """
            
            completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True
            )

            answer = ""
            for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                answer = answer + chunk.choices[0].delta.content.strip()
                # print(chunk.choices[0].delta.content, end="")

            answer_evaluation = answer_evaluation(model_answer, ground_truth)
            evals.append(answer_evaluation)


    elif data_type == "citations":
        for data_id in data_ids:
            prompt = load_prompt(data_type)
            cur.execute("""
            SELECT cited_arxiv_id, paper_arxiv_id FROM tables WHERE id = %s
            """, (data_id,))

            row = cur.fetchone()
            cited_arxiv_id, arxiv_id = None

            if row:
                cited_arxiv_id, arxiv_id = row

            # Remove the curl braces and signs from the bib_title
            # Assume that the paper with specified arxiv id exists in the database, fetch it

            cited_paper_title, cited_paper_abstract = cur.execute("""
            SELECT title, abstract FROM papers WHERE arxiv_id = %s
            """, (cited_arxiv_id))


            # First give the model the abstract of cited paper, then provide the model with each paragraphs.
            prompt = prompt + "Abstract of cited paper:\n"
            prompt = prompt + f"{{{cited_paper_abstract}}}\n"

            cur.execute("""
            SELECT id, context FROM paragraphs WHERE paper_arxiv_id = %s
            """, (arxiv_id))
            pairs = cur.fetchall()
            
            idx = 1
            prompt = prompt + "Paragraphs:\n"

            for paragraph_id, paragraph_context in pairs:
                # Give the context to model and see if it agrees on insertion (missing data imputation)
                
                paragraph_to_idx[paragraph_id] = idx
                # Append to a string?
                prompt = prompt + f"{idx}. {{{paragraph_context}}}\n"

                idx += 1
            

            prompt = prompt + """"
            ---

            Final Answer (only indexes, comma-separated):
            """
            

            completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True
            )

            answer = ""
            for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                answer = answer + chunk.choices[0].delta.content.strip()
                # print(chunk.choices[0].delta.content, end="")

            answer_evaluation = answer_evaluation(model_answer, ground_truth)
            evals.append(answer_evaluation)
    
    
    return evals

    # Query the related information



    # Method: iterative judgement over paragraphs. Go thourough each 



def paragraph_generation(args):
    
    """
    
    """
    
    pass

    