
from sentence_transformers import SentenceTransformer, util
from evaluate import load
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

load_dotenv()
API_KEY = os.getenv("API_KEY")

def rouge_similarity(generated_answers, original_answer):
    rouge = load("rouge")
    rouge_score = rouge.compute(predictions=[generated_answers], references=[original_answer])["rougeL"]
    return rouge_score

def sbert_similarity(generated_answers, original_answer):
    emb = model.encode([generated_answers, original_answer], normalize_embeddings=True)
    # With normalized embeddings, dot score == cosine similarity
    sim = util.dot_score(emb[0:1], emb[1:2]).item()      # in [-1, 1]
    return sim

def answer_evaluation(generated_answers, original_answers):
    """
    Evaluate answers as a list
    Assume that the two answer lists have the same length
    """

    result = []
    n_ans = len(generated_answers)

    for i in range(n_ans):
        generated_answer = generated_answers[i]
        original_answer = original_answers[i]
        rouge_score = rouge_similarity(generated_answer, original_answer)
        sbert_score = sbert_similarity(generated_answer, original_answer)
        result.append({"rouge_score": int(rouge_score), "sbert_score": sbert_score})
    
    return result
    
    

def gpt_evaluation(generated_answers, original_answer, model_name="nvdev/nvidia/llama-3.1-nemotron-70b-instruct"):
    RUBRIC = """TASK: Score the CANDIDATE against the REFERENCE.
    CRITERIA (1-5 each):
    - adequacy: semantic equivalence to REFERENCE
    - coverage: presence and order of key information from REFERENCE
    - fluency: grammar, clarity, coherence
    Return ONLY:
    {"adequacy": int, "coverage": int, "fluency": int, "overall": int,
    "notes": "≤2 sentences; no chain-of-thought."}"""

    user = f"{RUBRIC}\n\nREFERENCE:\n{original_answer}\n\nCANDIDATE:\n{generated_answers}"
    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = API_KEY
    )

    resp = client.chat.completions.create(
        model=model_name,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a strict writing evaluator. Return ONLY valid JSON."},
            {"role": "user", "content": user},
        ],
    )
    j = json.loads(resp.choices[0].message.content)
    return j
    