from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

def main():
    model_id = "Qwen/Qwen3-8B"  # or "Qwen/Qwen3-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(
        model_id,
        trust_remote_code=True,
        enforce_eager=True,          # avoid long torch.compile
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8, 
        dtype="float16",             # force FP16 instead of bf16
        # max_model_len=8192,        # uncomment to shrink KV cache further
        # disable_custom_all_reduce=True,  # optional stability toggle
    )
    out = llm.generate(["Hello from CUDA 12.4!"], SamplingParams(max_tokens=16, temperature=0))
    print("Output:")
    print(out[0].outputs[0].text)
    torch.cuda.empty_cache()
if __name__ == '__main__':    
    main()