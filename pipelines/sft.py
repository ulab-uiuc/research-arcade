from typing import List, Dict, Any, Optional
from tasks.paragraph_generation_local_vllm import llm_generate

class SFTPipeline:
    """
    Use a fine-tuned model checkpoint OR a LoRA adapter with the base model.
    - If `merged_model_or_lora_path` points to a full checkpoint, set `use_lora=False`.
    - If it points to a LoRA folder, set `use_lora=True`.
    Your `llm_generate` should pass these to vLLM (see note below).
    """
    def __init__(
        self,
        base_model_name: str,
        merged_model_or_lora_path: str,
        use_lora: bool = False,
        is_vlm: bool = False,
    ):
        self.base_model_name = base_model_name
        self.adapter_or_merged = merged_model_or_lora_path
        self.use_lora = use_lora
        self.is_vlm = is_vlm

    def run(
        self,
        paragraph_datas: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        original_contents: List[str] = []
        prompts: List[str] = []
        images_payload = []

        if self.is_vlm:
            for pd in paragraph_datas:
                original_contents.append(pd["original_content"])
                prompts.append(pd["prompt"])
                images_payload.append(pd["image_embeddings"])
        else:
            for pd in paragraph_datas:
                original_contents.append(pd["original_content"])
                prompts.append(pd["prompt"])
                images_payload.append(pd.get("image_tag_list", []))

        # IMPORTANT: update `llm_generate` to accept optional args:
        #   lora_path: Optional[str]
        #   use_lora: bool
        #   merged_model_path: Optional[str]
        #
        # vLLM options:
        #   (A) If you MERGED LoRA -> pass merged checkpoint via model_name (or path)
        #   (B) If you DID NOT merge -> keep base model, pass adapter via lora_path
        #
        gens = llm_generate(
            prompts=prompts,
            model_name=self.base_model_name if self.use_lora else self.adapter_or_merged,
            is_vlm=self.is_vlm,
            image_embeddings=images_payload if self.is_vlm else None,
            image_labels=None if self.is_vlm else images_payload,
            use_lora=self.use_lora,
            lora_path=self.adapter_or_merged if self.use_lora else None,
        )
        
        return {
            "prompts": prompts,
            "originals": original_contents,
            "generations": gens,
        }
