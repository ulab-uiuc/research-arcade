import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openai import OpenAI
import shutil
from graph_constructor.utils import figure_latex_path_to_path

from PIL import Image
import base64
from dotenv import load_dotenv
import sys
import fitz
import io

load_dotenv()

API_KEY = os.getenv("API_KEY")


download_path = "./download"
paper_arxiv_id = "2008.03299"
latex_path = "Konigsberg2.pdf"
fs_path = figure_latex_path_to_path(
    path=download_path,
    arxiv_id=paper_arxiv_id,
    latex_path=latex_path
)
if not os.path.isfile(fs_path):
    print("File cannot be opened")

if fs_path.endswith(".pdf"):
    doc = fitz.open(fs_path)
    for page_num in range(len(doc)):
        page = doc[page_num]

        # Step 2: Render PDF page to an image (pixmap)
        pix = page.get_pixmap()

        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Step 3: Save image to memory buffer (JPEG format)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")

        # Step 4: Encode in base64
        image_b64_0 = base64.b64encode(buffer.getvalue()).decode("utf-8")

else:
    with open(fs_path, 'rb') as f:
        image_b64_0 = base64.b64encode(f.read()).decode('utf-8')


# image_data_url = f"data:application/pdf;base64,{img_b64}"

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = API_KEY
)

def generate_answer(text):
    completion = client.chat.completions.create(
      # model="nvdev/meta/llama-3.1-405b-instruct",
      model="nvdev/nvidia/llama-3.1-nemotron-70b-instruct",
      messages=[{"role":"user","content":text,}],
      temperature=0.2,
      top_p=0.7,
      max_tokens=8192,
      stream=True
    )
    
    for chunk in completion:
      if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

def generate_answer_with_figure(text, figure_embedding):
    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        messages=[
            {
                "role": "user",
                "content": [
                { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{figure_embedding}" } },
                { "type": "text", "text": text }
                ]
            }
            ],
        temperature=1.00,
        top_p=0.01,
        max_tokens=1024,
        stream=True
    )
        
    for chunk in completion:
      if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
    print("")



generate_answer_with_figure(text = "Describe the image presented, detailed depict what is in the image. Make a guess about where the image comes from.", figure_embedding =image_b64_0)
# Make a copy of it
