from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama-3.1-405b-instruct",
  messages=[{"role":"user","content":"Write a limerick about the wonders of GPU computing."}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")


client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = ""
)

def generate_answer(text):
    completion = client.chat.completions.create(
      # model="nvdev/meta/llama-3.1-405b-instruct",
      model="nvdev/nvidia/llama-3.1-nemotron-70b-instruct",
      messages=[{"role":"user","content":text}],
      temperature=0.2,
      top_p=0.7,
      max_tokens=8192,
      stream=True
    )
    
    for chunk in completion:
      if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
