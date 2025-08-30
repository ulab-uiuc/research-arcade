"""
Perform supervised fine tuning given the input data

Input: prompt + image embedding/tags if provided
Output: generated paragraphs

Supervise label: original paragraphs
"""
import os
import sys
from typing import List, Dict, Any, Optional

# --- make sure we can import your module ---
# Edit this path if your file lives elsewhere
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from paragraph_generation_local_vllm import _data_extraction_non_vlm  # ← your existing method

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


# ===============================
# Config (edit to your setup)
# ===============================
TRAIN_JSONL = "train.jsonl"
EVAL_JSONL  = "eval.jsonl"     # optional but recommended

# Good Qwen base choices:
#   "Qwen/Qwen3-8B-Instruct" (needs ≥1 decent GPU; use 4-bit to fit)
#   "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "Qwen/Qwen3-8B-Instruct"

OUTPUT_DIR = "./sft-qwen-paragraph"
USE_4BIT = True
BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

SYSTEM_PROMPT = (
    "You are a careful scientific writer who produces one valid LaTeX paragraph "
    "that logically fits the provided context and exactly follows the citation/figure/table rules."
)

# LoRA target modules—common for Qwen/LLaMA families. Adjust if your model differs.
LORA_CFG = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

SFT_ARGS = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_seq_length=2048,
    packing=False,  # variable-length paragraphs; keep examples separate
    bf16=BF16,
    fp16=not BF16,
    gradient_checkpointing=True,
    report_to="none",
)


# ===============================
# Helpers: format text for SFT
# ===============================
def _build_input_text(prompt: str, image_tags: Optional[List[str]]) -> str:
    """
    We keep a simple instruction-style layout so it works across Qwen variants.
    If you want to use Qwen's chat template, you can wrap with <|im_start|>... later.
    """
    tags_block = ""
    if image_tags:
        tags_block = "\n<image_tags>" + ", ".join(map(str, image_tags[:20])) + "</image_tags>"
    
    return (
        f"### System\n{SYSTEM_PROMPT}\n\n"
        f"### User\n{prompt}{tags_block}\n\n"
        f"### Assistant\n"
    )

def _format_for_sft(rec: Dict[str, Any]) -> Dict[str, str]:
    """
    Your _data_extraction_non_vlm returns:
      - 'prompt': str
      - 'original_content': str
      - 'image_tag_list': List[str]  (derived via CLIP or precomputed)
    We combine into a single 'text' = input + target for SFTTrainer.
    """
    prompt = (rec.get("prompt") or "").strip()
    target = (rec.get("original_content") or "").strip()
    tags   = rec.get("image_tag_list") or []
    src = _build_input_text(prompt, tags)
    return {"text": src + target}


# ===============================
# Build HF datasets from your method
# ===============================
def build_datasets(train_jsonl: str, eval_jsonl: Optional[str] = None) -> DatasetDict:
    train_records = _data_extraction_non_vlm(train_jsonl)
    train_ds = Dataset.from_list([_format_for_sft(r) for r in train_records])

    data = {"train": train_ds}

    if eval_jsonl and os.path.exists(eval_jsonl):
        eval_records = _data_extraction_non_vlm(eval_jsonl)
        eval_ds = Dataset.from_list([_format_for_sft(r) for r in eval_records])
        data["validation"] = eval_ds

    return DatasetDict(data)


# ===============================
# Main training
# ===============================
def main():
    # BitsAndBytes 4-bit (QLoRA)
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

    # If your tokenizer lacks these markers, it's fine—we're not relying on special IDs.
    # (We purposely avoid model-specific chat tokens to keep things robust.)
    # But you could add special tokens here if you standardize on them.

    model_kwargs = dict(trust_remote_code=True)
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)


    # Load data using your own extraction pipeline
    print("Preparing datasets via your _data_extraction_non_vlm...")
    sys.exit()
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

    # Optional: merge LoRA into base weights for an all-in-one fp16/bf16 model
    # (commented out by default because it's VRAM-heavy).
    # from peft import PeftModel
    # merged = PeftModel.from_pretrained(model, OUTPUT_DIR).merge_and_unload()
    # merged.save_pretrained(f"{OUTPUT_DIR}-merged")

    # Quick generation sanity check (HF, not vLLM)
    if torch.cuda.is_available():
        model.to("cuda")
    gen = GenerationConfig(
        max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9,
        repetition_penalty=1.05, eos_token_id=tok.eos_token_id
    )

    demo_prompt = (
        "Given section=Experiments, summarize comparative results with baselines, "
        "and mention the ablation highlights."
    )
    demo_tags = ["ablation study", "imagenet", "baseline comparison"]

    inp = _build_input_text(demo_prompt, demo_tags)
    inputs = tok(inp, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen)
    text = tok.decode(out[0], skip_special_tokens=False)
    if "### Assistant" in text:
        text = text.split("### Assistant")[-1]
    print("\n=== SAMPLE OUTPUT ===\n", text.strip())


if __name__ == "__main__":
    main()
