"""
paragraph_generation
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Callable, List, Optional, Tuple, Dict

from dataclasses import dataclass
from tasks.paragraph_generation import paragraph_generation
from tasks.generated_paragraph_evaluation import rouge_similarity, sbert_similarity, gpt_evaluation

@dataclass
class Args:
    paragraph_ids: List[int]
    model_name: str
    k_neighbour: int = 2
    figure_available: bool = True
    table_available: bool = True
    download_path: str = "./download"

if __name__ == "__main__":

    args = Args(
        paragraph_ids=[21812255],
        k_neighbour=4,
        figure_available=True,
        table_available=True,
        model_name="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        download_path="./download"
    )
    
    results = paragraph_generation(args)
    
    # for r in results:
    #     print("\n=== Paragraph ID:", r["paragraph_id"], "===\n")
    #     print("PROMPT:\n", r["prompt"])
    #     if r["llm_output"]:
    #         print("\nLLM OUTPUT:\n", r["llm_output"])
    
    # Evaluate the generated results

    for result in results:
        prompt = result["prompt"]
        generate_answer = result["llm_output"]
        original_answer = result["original content"]
        print(f"prompt:{prompt}")
        print("---------------------------------------------------")
        print(f"generate_answer: {generate_answer}")
        print("---------------------------------------------------")
        print(f"original_answer: {original_answer}")
        print("---------------------------------------------------")
        
        #TODO remove it. This is just for skipping the evals
        # continue
        rouge_score = rouge_similarity(generate_answer, original_answer)
        print(f"Rouge Score: {rouge_score}")
        sbert_score = sbert_similarity(generate_answer, original_answer)
        print(f"SBERT Score: {sbert_score}")
        gpt_evaluation_score = gpt_evaluation(generate_answer, original_answer)
    
        print(f"GPT Evaluation Score: {gpt_evaluation_score}")
        print("---------------------------------------------------")

    #     outputs.append({
    #         "paragraph_id": str(pid),
    #         "prompt": prompt,
    #         "original content": paragraph_content,
    #         "llm_output": llm_output or ""
    #     })
