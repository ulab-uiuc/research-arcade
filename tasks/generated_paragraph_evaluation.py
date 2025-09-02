
from sentence_transformers import SentenceTransformer, util
from evaluate import load
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict, Any, Optional
from evaluate import load as hf_load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

load_dotenv()
API_KEY = os.getenv("API_KEY")
_smooth = SmoothingFunction().method4

def rouge_similarity_batch(
    preds: List[str],
    refs: List[str],
    rouge_metric=None,
    rouge_type: str = "rougeL"
) -> List[float]:
    """
    Returns a list of ROUGE scores aligned with input pairs.
    HuggingFace 'rouge' computes corpus-level by default; but it
    also returns per-example scores if you pass lists—access via 'precisions/recalls/fmeasures'
    is not exposed; so we call it pairwise in a lightweight loop to get per-pair.
    """
    # For true batch speed, you can keep it corpus-level. Here we keep per-pair parity with your API.
    if rouge_metric is None:
        rouge_metric = hf_load("rouge")
    scores = []
    for p, r in zip(preds, refs):
        out = rouge_metric.compute(predictions=[p], references=[r])
        scores.append(out.get(rouge_type, 0.0))
    return scores

def sbert_similarity_batch(
    model: SentenceTransformer,
    preds: List[str],
    refs: List[str],
) -> List[float]:
    """
    Batched SBERT cosine similarity for aligned pairs.
    """
    emb_preds = model.encode(preds, normalize_embeddings=True, convert_to_tensor=True)
    emb_refs  = model.encode(refs,  normalize_embeddings=True, convert_to_tensor=True)
    # With normalized embeddings, cosine == dot product
    sims = (emb_preds * emb_refs).sum(dim=1)  # [N]
    return sims.detach().cpu().tolist()

def bleu_similarity_batch(
    preds: List[str],
    refs: List[str],
    weights=(0.25, 0.25, 0.25, 0.25),
) -> List[float]:
    """
    Per-pair BLEU-4 with smoothing (method4). Tokenizes by whitespace.
    """
    # Pre-tokenize
    ref_tok = [r.split() for r in refs]
    pred_tok = [p.split() for p in preds]
    scores = [
        sentence_bleu([rt], pt, weights=weights, smoothing_function=_smooth) if pt else 0.0
        for rt, pt in zip(ref_tok, pred_tok)
    ]
    return scores

def answer_evaluation_batch(
    generated_answers: List[str],
    original_answers: List[str],
    model: SentenceTransformer,
    rouge_metric=None
) -> Dict[str, Any]:
    """
    Batched evaluation for aligned lists of answers.
    Returns:
      {
        "per_item": [{"rouge": ..., "sbert": ..., "bleu": ...}, ...],
        "averages": {"rouge": ..., "sbert": ..., "bleu": ...}   # if return_averages
      }
    """
    assert len(generated_answers) == len(original_answers), "Lists must be same length."

    # Reuse a single rouge metric instance if provided
    rouge_scores = rouge_similarity_batch(generated_answers, original_answers, rouge_metric, "rougeL")
    sbert_scores = sbert_similarity_batch(model, generated_answers, original_answers)
    bleu_scores  = bleu_similarity_batch(generated_answers, original_answers)

    per_item = [
        {"rouge_score": r, "sbert_score": s, "bleu_score": b}
        for r, s, b in zip(rouge_scores, sbert_scores, bleu_scores)
    ]
    
    return per_item

def rouge_similarity(generated_answers, original_answer):
    rouge = load("rouge")
    rouge_score = rouge.compute(predictions=[generated_answers], references=[original_answer])["rougeL"]
    return rouge_score

def sbert_similarity(generated_answers, original_answer):
    emb = model.encode([generated_answers, original_answer], normalize_embeddings=True)
    # With normalized embeddings, dot score == cosine similarity
    sim = util.dot_score(emb[0:1], emb[1:2]).item()      # in [-1, 1]
    return sim

def bleu_similarity(generated_answer, original_answer):
    """
    Compute BLEU score between generated text and reference text.
    Returns a float in [0,1].
    """
    # Tokenize by whitespace; you can swap for a real tokenizer if needed
    reference = original_answer.split()
    candidate = generated_answer.split()

    # BLEU expects a list of references (each reference is a list of tokens)
    references = [reference]
    candidate_tokens = candidate

    # Smoothing is critical for long texts or when n-gram overlap is sparse
    smoothie = SmoothingFunction().method4

    score = sentence_bleu(
        references,
        candidate_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),  # BLEU-4
        smoothing_function=smoothie
    )

    return score

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
        bleu_score = bleu_similarity(generated_answer, original_answer)

        result.append({"rouge_score": rouge_score, "sbert_score": sbert_score, "bleu_score": bleu_score})
        
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
    