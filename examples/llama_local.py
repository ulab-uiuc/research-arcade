from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

path = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
model = AutoModel.from_pretrained(path, trust_remote_code=True, device_map="cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(path)
image_processor = AutoImageProcessor.from_pretrained(path, trust_remote_code=True, device="cuda")

image1 = Image.open("images/example1a.jpeg")
image2 = Image.open("images/example1b.jpeg")
image_features = image_processor([image1, image2])

generation_config = dict(max_new_tokens=1024, do_sample=False, eos_token_id=tokenizer.eos_token_id)

question = 'Describe the two images.'
response = model.chat(
    tokenizer=tokenizer, question=question, generation_config=generation_config,
    **image_features)

print(f'User: {question}\nAssistant: {response}')
