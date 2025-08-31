from typing import List, Dict, Any
from tasks.paragraph_generation_local_vllm import llm_generate

class ZeroShotPipeline:
    def __init__(self, model_name: str, is_vlm: bool):
        self.model_name = model_name
        self.is_vlm = is_vlm

    def run(
        self,
        paragraph_datas: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        original_contents: List[str] = []
        prompts: List[str] = []
        images_payload = []

        if self.is_vlm:
            # VLM path: collect image embeddings
            for pd in paragraph_datas:
                original_contents.append(pd["original_content"])
                prompts.append(pd["prompt"])
                images_payload.append(pd["image_embeddings"])
            gens = llm_generate(
                prompts=prompts,
                model_name=self.model_name,
                is_vlm=True,
                image_embeddings=images_payload,
            )
        else:
            # Non-VLM path: collect image tag lists (labels)
            for pd in paragraph_datas:
                original_contents.append(pd["original_content"])
                prompts.append(pd["prompt"])
                images_payload.append(pd.get("image_tag_list", []))
            gens = llm_generate(
                prompts=prompts,
                model_name=self.model_name,
                is_vlm=False,
                image_labels=images_payload,
            )
        
        return {
            "prompts": prompts,
            "originals": original_contents,
            "generations": gens,
        }
