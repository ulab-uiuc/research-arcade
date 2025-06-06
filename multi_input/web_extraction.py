import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

# URL of the arXiv HTML page
url = "https://arxiv.org/html/2503.12600"

# Fetch HTML content
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 1. Extract and save all text
text = soup.get_text()
with open("grapheval_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

# 2. Extract all image URLs and download them
image_tags = soup.find_all("img")
image_urls = [urljoin(url, img['src']) for img in image_tags if 'src' in img.attrs]

os.makedirs("images", exist_ok=True)
for idx, img_url in enumerate(image_urls):
    try:
        img_data = requests.get(img_url).content
        with open(f"images/image_{idx + 1}.png", "wb") as img_file:
            img_file.write(img_data)
    except Exception as e:
        print(f"Could not download {img_url}: {e}")

# 3. Extract all hyperlinks
link_tags = soup.find_all("a")
links = [urljoin(url, link['href']) for link in link_tags if 'href' in link.attrs]

with open("grapheval_links.txt", "w", encoding="utf-8") as f:
    for link in links:
        f.write(link + "\n")
