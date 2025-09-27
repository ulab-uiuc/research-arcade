import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

class MultiModalQwenLoader:
    def __init__(self, load_directory: str, base_model_name: str = "Qwen/Qwen3-0.6B", device: str = "cuda"):
        self.load_directory = load_directory
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load tokenizer (with your extra special tokens already saved)
        self.tokenizer = AutoTokenizer.from_pretrained(load_directory)

        # Load config file (saved in save_pretrained)
        with open(os.path.join(load_directory, "model_config.json"), "r") as f:
            config = json.load(f)

        embedding_dim = config["embedding_dim"]
        hidden_size = config["hidden_size"]
        use_lora = config["use_lora"]

        # Step 1: Reload the *base model* from hub, not from the saved LoRA dir
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        # Step 2: If LoRA was used, wrap and load adapter weights
        if use_lora:
            self.base_model = PeftModel.from_pretrained(self.base_model, load_directory)
        self.base_model.to(self.device)

        # Step 3: Load projector
        projector_path = os.path.join(load_directory, "projector_weights.bin")
        self.embedding_projector = torch.nn.Linear(embedding_dim, hidden_size)
        projector_state = torch.load(projector_path, map_location="cpu")
        self.embedding_projector.load_state_dict(projector_state["embedding_projector"])
        self.embedding_projector.to(self.device)

        print(f"✅ Model + projector loaded successfully from {load_directory}")

    def get_model(self):
        return self.base_model, self.embedding_projector

    def get_tokenizer(self):
        return self.tokenizer

save_dir = "./data/paragraph_generation/final_paragraph_generation_model"

loader = MultiModalQwenLoader(save_dir)
model, projector = loader.get_model()
tokenizer = loader.get_tokenizer()

# Quick sanity check
text = "Hello world!"
ids = tokenizer.encode(text, return_tensors="pt").to(loader.device)
outputs = model.generate(ids, max_new_tokens=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
