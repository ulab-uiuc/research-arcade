import os, json
from dataclasses import dataclass
from typing import Dict
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoModelForCausalLM, TrainingArguments

"""
Assumes your JSONL rows look like:
{
  "prompt": "…",                # what you fed to the model
  "original_content": "…"       # gold paragraph to train against (target)
  // optional: "image_tag_list": [...],  "image_embeddings": ...
}
You can always enrich the prompt beforehand with tags if desired.
"""

@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen3-8B"
    jsonl_path: str = "./jsonl/paragraph_generation2.jsonl"
    output_dir: str = "./sft_output/qwen3-8b-qora"
    max_seq_len: int = 4096
    micro_batch_size: int = 2
    grad_accum_steps: int = 8
    lr: float = 2e-4
    epochs: int = 3
    fp8: bool = False  # keep False unless you know your stack supports it

def load_jsonl_as_pairs(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            prompt = j["prompt"]
            target = j["original_content"]
            # Optionally inject image tags into the prompt:
            tags = j.get("image_tag_list")
            if tags:
                prompt = f"{prompt}\n\n[IMAGE_TAGS]: {', '.join(tags)}"
            records.append({"text": f"{prompt}\n\n### Answer:\n{target}"})
    return Dataset.from_list(records)

def main(cfg: TrainConfig = TrainConfig()):
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 4-bit quantization (QLoRA)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
    )

    # Base model (frozen) + LoRA on top
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb,
        trust_remote_code=True,
        device_map="auto",
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],  # adjust per arch
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_ds = load_jsonl_as_pairs(cfg.jsonl_path)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        peft_config=lora,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_len,
        packing=True,  # efficient packing of multiple examples
        args=TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.micro_batch_size,
            gradient_accumulation_steps=cfg.grad_accum_steps,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            report_to="none",
        ),
    )

    trainer.train()
    # Save the LoRA adapter (this is small)
    trainer.model.save_pretrained(cfg.output_dir)
    trainer.tokenizer.save_pretrained(cfg.output_dir)

    # OPTIONAL: Merge LoRA into the base weights (for easy vLLM use without LoRA)
    try:
        from peft import AutoPeftModelForCausalLM
        merged = AutoPeftModelForCausalLM.from_pretrained(
            cfg.output_dir,
            device_map="auto",
            trust_remote_code=True
        )
        merged = merged.merge_and_unload()  # applies LoRA into base
        merged_dir = os.path.join(cfg.output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        print(f"Merged checkpoint saved to: {merged_dir}")
    except Exception as e:
        print("Merge step skipped (you can still use LoRA adapter):", e)

if __name__ == "__main__":
    main()
