import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, argparse, csv, json
from typing import List, Dict, Any
from tasks.paragraph_generation_local_vllm import _data_extraction_vlm, _data_extraction_non_vlm
from tasks.generated_paragraph_evaluation import answer_evaluation
from pipelines.zero_shot import ZeroShotPipeline
from pipelines.sft import SFTPipeline

def load_paragraph_datas(jsonl_path: str, model_name: str, vlm: bool) -> List[Dict[str, Any]]:
    if vlm:
        return _data_extraction_vlm(jsonl_file_path=jsonl_path, model_name=model_name)
    else:
        return _data_extraction_non_vlm(jsonl_file_path=jsonl_path, model_name=model_name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=["zero_shot","sft"], required=True)
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--jsonl", default="./jsonl/paragraph_generation2.jsonl")
    ap.add_argument("--base_model", default="Qwen/Qwen3-8B")
    ap.add_argument("--sft_path", default="./sft_output/qwen3-8b-qora")   # adapter or merged
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--result_csv", default="./task_result/paragraph_generation_result2.csv")
    ap.add_argument("--paragraph_ids", nargs="+", default=["21829759","21831811","21854471","56348","2848716","21846899"])
    args = ap.parse_args()

    paragraph_datas = load_paragraph_datas(args.jsonl, args.base_model, args.vlm)

    if args.pipeline == "zero_shot":
        pipe = ZeroShotPipeline(model_name=args.base_model, is_vlm=args.vlm)
    else:
        # SFT inference: if use_lora -> base_model + adapter; else -> merged checkpoint as model_name
        pipe = SFTPipeline(
            base_model_name=args.base_model,
            merged_model_or_lora_path=args.sft_path,
            use_lora=args.use_lora,
            is_vlm=args.vlm,
        )

    out = pipe.run(paragraph_datas)
    prompts = out["prompts"]
    originals = out["originals"]
    generations = out["generations"]

    # Evaluate
    evals = answer_evaluation(generations, originals)

    # Save CSV
    os.makedirs(os.path.dirname(args.result_csv), exist_ok=True)
    fieldnames = ["paragraph_id","prompt","original_content","generated_paragraph"]
    eval_keys = set()
    for e in evals:
        if isinstance(e, dict): eval_keys.update(e.keys())
    fieldnames.extend(sorted(eval_keys))

    with open(args.result_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for pid, pr, orig, gen, ev in zip(args.paragraph_ids, prompts, originals, generations, evals):
            row = {
                "paragraph_id": pid,
                "prompt": pr,
                "original_content": orig,
                "generated_paragraph": gen,
            }
            if isinstance(ev, dict): row.update(ev)
            wr.writerow(row)

    print(f"Saved results to {args.result_csv}")

if __name__ == "__main__":
    main()
