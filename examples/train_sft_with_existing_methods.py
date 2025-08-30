# -*- coding: utf-8 -*-
import os, sys, json, random
from typing import List, Dict, Any, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tasks.paragraph_generation_local_vllm import _data_extraction_non_vlm  # YOUR extractor

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# --------- Paths (edit) ----------
# TRAIN_JSONL = "./jsonl/paragraph_generation2.jsonl"
# EVAL_JSONL  = "./jsonl/paragraph_generation2_eval.jsonl"   # if missing, we will auto-split
# VAL_RATIO   = 0.10  # used only when EVAL_JSONL doesn't exist

# --------- Model ----------
MODEL_NAME  = "Qwen/Qwen3-8B-Instruct"
OUTPUT_DIR  = "./sft-qwen-paragraph"
USE_4BIT    = True
SEED        = 42
BF16        = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

LORA_CFG = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

SFT_ARGS = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    max_seq_length=2048,
    packing=False,
    bf16=BF16,
    fp16=not BF16,
    gradient_checkpointing=True,
    report_to="none",
    seed=SEED,
)

# --------- Formatting (mirror your llm_generate preface) ----------
def _tags_preface(tags: Optional[List[str]]) -> str:
    if not tags:
        return ""
    return "You may use the following image tag summary as auxiliary context:\n" \
           f"[Image tags]: {', '.join(map(str, tags[:32]))}\n\n"

def _format_text_for_sft(rec: Dict[str, Any]) -> Dict[str, str]:
    prompt = (rec.get("prompt") or "").strip()
    target = (rec.get("original_content") or "").strip()
    tags   = rec.get("image_tag_list") or []
    text = f"{_tags_preface(tags)}{prompt}\n{target}"
    return {"text": text}

# --------- Data utils ----------
def _load_jsonl_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln for ln in (l.strip() for l in f) if ln]

def _save_jsonl_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def ensure_train_eval_files(train_path: str, eval_path: str, val_ratio: float, seed: int) -> Tuple[str, Optional[str]]:
    """If eval_path exists, use it. Otherwise, split train_path into train/eval JSONLs and return new paths."""
    if os.path.exists(eval_path):
        return train_path, eval_path

    lines = _load_jsonl_lines(train_path)
    if len(lines) < 2:
        raise ValueError("Not enough examples to create a validation split.")
    rnd = random.Random(seed)
    rnd.shuffle(lines)
    n_val = max(1, int(round(len(lines) * val_ratio)))
    val_lines = lines[:n_val]
    trn_lines = lines[n_val:]

    base, ext = os.path.splitext(train_path)
    new_train = f"{base}.train{ext or '.jsonl'}"
    new_eval  = f"{base}.eval{ext or '.jsonl'}"
    _save_jsonl_lines(new_train, trn_lines)
    _save_jsonl_lines(new_eval,  val_lines)

    print(f"[Auto-split] Wrote {len(trn_lines)} train lines → {new_train}")
    print(f"[Auto-split] Wrote {len(val_lines)}  eval  lines → {new_eval}")
    return new_train, new_eval

def build_datasets(train_jsonl: str, eval_jsonl: Optional[str]) -> DatasetDict:
    train_records = _data_extraction_non_vlm(train_jsonl)
    train_ds = Dataset.from_list([_format_text_for_sft(r) for r in train_records])

    data = {"train": train_ds}
    if eval_jsonl and os.path.exists(eval_jsonl):
        eval_records = _data_extraction_non_vlm(eval_jsonl)
        eval_ds = Dataset.from_list([_format_text_for_sft(r) for r in eval_records])
        data["validation"] = eval_ds
    return DatasetDict(data)

# --------- Main ----------
def main():
    # Create/locate validation set
    tr_path, ev_path = ensure_train_eval_files(TRAIN_JSONL, EVAL_JSONL, VAL_RATIO, SEED)

    # Quantization
    quant_cfg = None
    if USE_4BIT:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
        )

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(trust_remote_code=True)
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

    # Build datasets via your extractor (derives tags if needed)
    dsd = build_datasets(tr_path, ev_path)
    print(dsd)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=dsd["train"],
        eval_dataset=dsd.get("validation"),
        peft_config=LORA_CFG,
        args=SFT_ARGS,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

    # Final eval: report eval loss + perplexity if we had a validation set
    if "validation" in dsd:
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            try:
                import math
                ppl = math.exp(eval_loss)
                print(f"[Final Eval] loss={eval_loss:.4f} | ppl={ppl:.2f}")
            except Exception:
                pass

    # Smoke test
    if torch.cuda.is_available():
        model.to("cuda")
    gc = GenerationConfig(max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9,
                          repetition_penalty=1.05, eos_token_id=tok.eos_token_id)
    demo = "Given section=Experiments, summarize comparative results and mention ablation highlights."
    demo_tags = ["ablation study", "imagenet", "baseline comparison"]
    demo_inp = f"{_tags_preface(demo_tags)}{demo}\n"
    inputs = tok(demo_inp, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gc)
    print("\n=== SAMPLE OUTPUT (tail) ===\n", tok.decode(out[0], skip_special_tokens=False)[-1000:].strip())

if __name__ == "__main__":
    main()
