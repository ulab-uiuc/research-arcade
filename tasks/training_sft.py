# -*- coding: utf-8 -*-
"""
Supervised fine-tuning using your existing extraction & prompt pipeline.

- Uses _data_extraction_non_vlm(...) to create text-only training examples.
- Mirrors the tag-preface used in llm_generate() so train/infer match.
- Trains a Qwen family causal LM with QLoRA via TRL's SFTTrainer.
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Make sure we can import your tasks module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.paragraph_generation_local_vllm import _data_extraction_non_vlm  # ← YOURS

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# ---------- Config (edit) ----------
TRAIN_JSONL = "./jsonl/paragraph_generation2.jsonl"
EVAL_JSONL  = "./jsonl/paragraph_generation2_eval.jsonl"   # optional

MODEL_NAME  = "Qwen/Qwen3-8B-Instruct"   # works well; keep 4-bit if VRAM is tight
OUTPUT_DIR  = "./sft-qwen-paragraph"
USE_4BIT    = True
SEED        = 42

BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

# LoRA for Qwen/LLaMA style blocks
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
    packing=False,  # variable-length; keep examples separate
    bf16=BF16,
    fp16=not BF16,
    gradient_checkpointing=True,
    report_to="none",
    seed=SEED,
)


# ---------- Formatting to mirror your llm_generate() ----------
def _tags_preface(tags: Optional[List[str]]) -> str:
    if not tags:
        return ""
    tag_text = ", ".join(map(str, tags[:32]))
    return (
        "You may use the following image tag summary as auxiliary context:\n"
        f"[Image tags]: {tag_text}\n\n"
    )

def _format_text_for_sft(rec: Dict[str, Any]) -> Dict[str, str]:
    """
    _data_extraction_non_vlm output (relevant keys):
      - prompt: str                  (already built by your _build_prompt)
      - original_content: str
      - image_tag_list: List[str]    (CLIP-derived)
    We concatenate: TAG_PREFACE + PROMPT + TARGET
    """
    prompt = (rec.get("prompt") or "").strip()
    target = (rec.get("original_content") or "").strip()
    tags   = rec.get("image_tag_list") or []

    text = f"{_tags_preface(tags)}{prompt}\n{target}"
    return {"text": text}


# ---------- Build DatasetDict from your extractor ----------
def build_datasets(train_jsonl: str, eval_jsonl: Optional[str]) -> DatasetDict:
    train_records = _data_extraction_non_vlm(train_jsonl)
    train_ds = Dataset.from_list([_format_text_for_sft(r) for r in train_records])

    data = {"train": train_ds}

    if eval_jsonl and os.path.exists(eval_jsonl):
        eval_records = _data_extraction_non_vlm(eval_jsonl)
        eval_ds = Dataset.from_list([_format_text_for_sft(r) for r in eval_records])
        data["validation"] = eval_ds

    return DatasetDict(data)


def main():
    # Quantization (QLoRA)
    quant_cfg = None
    if USE_4BIT:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
        )

    print("Loading tokenizer/model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
    # Pad token safety for causal LMs
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(trust_remote_code=True)
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

    print("Building datasets using your extractor...")
    dsd = build_datasets(TRAIN_JSONL, EVAL_JSONL if os.path.exists(EVAL_JSONL) else None)
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

    # Quick smoke test (Transformers path)
    if torch.cuda.is_available():
        model.to("cuda")

    gen_cfg = GenerationConfig(
        max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9,
        repetition_penalty=1.05, eos_token_id=tok.eos_token_id
    )

    demo_prompt = "Given section=Experiments, summarize comparative results, mention ablation highlights."
    demo_tags   = ["ablation study", "imagenet", "baseline comparison"]

    demo_input = f"{_tags_preface(demo_tags)}{demo_prompt}\n"
    inputs = tok(demo_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    text = tok.decode(out[0], skip_special_tokens=False)
    print("\n=== SAMPLE OUTPUT (truncated) ===\n", text[-1000:].strip())


if __name__ == "__main__":
    main()
