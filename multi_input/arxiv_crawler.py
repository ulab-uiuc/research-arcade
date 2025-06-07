import os
import requests
import time
import xml.etree.ElementTree as ET
import tarfile
import shutil
# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

# (1) Folder where PDFs, source .tar.gz, and metadata files will be saved
DOWNLOAD_DIR = "arxiv_papers_with_source"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

POSTFIX = ".tar.gz"

# (2) How many entries to fetch per API call (50 is a polite default)
ENTRIES_PER_CALL = 50

# (3) Seconds to pause between successive API calls (avoid hammering arXiv)
PAUSE_BETWEEN_CALLS = 3

# (4) Base endpoint for arXiv’s Atom API
BASE_URL = "http://export.arxiv.org/api/query"

# (5) Base URL for downloading source tarballs
SOURCE_BASE_URL = "https://arxiv.org/e-print/"

# ─── DATE-RANGE / CATEGORY SETUP ────────────────────────────────────────────────

# Define your date range here (UTC, in YYYYMMDDhhmm format).
# Example: Jan 1 2025 00:00 to May 31 2025 23:59
start_date = "202501010000"
end_date   = "202505312359"

# Choose your subject/category or free‐text as usual (e.g., "cs.AI")
category = "cs.AI"

# Build the arXiv search_query string:
#   “cat:cs.AI AND submittedDate:[202501010000 TO 202505312359]”
search_query = f"cat:{category} AND submittedDate:[{start_date} TO {end_date}]"

# ─── FETCH A SINGLE BATCH ───────────────────────────────────────────────────────

def fetch_arxiv_batch(search_query, start=0, max_results=ENTRIES_PER_CALL):
    """
    Hit arXiv's API with given search_query, start index, and max_results.
    Returns raw Atom-XML text.
    """
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.text


# ─── PARSE ATOM FEED INTO A LIST OF DICTS ───────────────────────────────────────

def parse_arxiv_feed(xml_data):
    # """
    # Given Atom XML data, parse each <entry> and return a list of dicts:
    # - id, title, summary, published, authors[], pdf_link
    # """
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    papers = []

    for entry in root.findall("atom:entry", ns):
        paper = {
            "id": entry.find("atom:id", ns).text.strip(),
            "title": entry.find("atom:title", ns).text.strip(),
            "summary": entry.find("atom:summary", ns).text.strip(),
            "published": entry.find("atom:published", ns).text.strip(),
        }

        # AUTHORS
        authors = []
        for au in entry.findall("atom:author", ns):
            name = au.find("atom:name", ns).text.strip()
            authors.append(name)
        paper["authors"] = authors

        # PDF LINK (usually <link title="pdf" href="…">)
        pdf_link = None
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_link = link.get("href")
                break
        paper["pdf_link"] = pdf_link

        papers.append(paper)

    return papers


def download_pdf(pdf_url, dest_path):
    """
    Stream the PDF at pdf_url and write it to dest_path.
    """
    resp = requests.get(pdf_url, stream=True)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def download_source(arxiv_id, dest_path):
    """
    Download the LaTeX source tarball for a given arXiv ID.
    """
    source_url = SOURCE_BASE_URL + arxiv_id
    resp = requests.get(source_url, stream=True)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def extract_tar_gz_files(directory):
    """Extracts all tar.gz files in the specified directory."""
    # for filename in os.listdir(directory):
    #     if filename.endswith(".tar.gz"):
    #         file_path = os.path.join(directory, filename)
    #         try:
    #             with tarfile.open(file_path, "r:gz") as tar:
    #                 tar.extractall(path=directory)
    #             print(f"Extracted: {filename}")
    #         except tarfile.ReadError:
    #             print(f"Error: Could not read {filename}. File might be corrupted or not a valid tar.gz file.")
    #         except Exception as e:
    #              print(f"Error extracting {filename}: {e}")
    for filename in os.listdir(directory):
        if filename.endswith(POSTFIX):
            file_path = os.path.join(directory, filename)
            print(filename[0: -len(POSTFIX)])
            destination_dir = os.path.join(directory, filename[0: -len(POSTFIX)])
            try:
                shutil.unpack_archive(filename=file_path, extract_dir=destination_dir)
                # with tarfile.open(file_path, "r:gz") as tar:
                #     tar.extractall(path=file_path)
                # print(f"Extracted: {filename}")
            except shutil.ReadError:
                print(f"Error: Could not read {filename}. File might be corrupted or not a valid tar.gz file.")
            except Exception as e:
                print(f"Error extracting {filename}: {e}")
                # print(type(e))
        


def crawl_arxiv_with_source(search_query, max_results=100):
    """
    Crawl up to max_results papers from arXiv, restricted by search_query
    (which may include a date range). For each paper:
    1. Download PDF (if available)
    2. Download LaTeX source (.tar.gz)
    3. Save a metadata .txt alongside them.
    """
    start = 0
    downloaded = 0

    while downloaded < max_results:
        batch_size = min(ENTRIES_PER_CALL, max_results - downloaded)
        xml_data = fetch_arxiv_batch(search_query, start=start, max_results=batch_size)
        batch = parse_arxiv_feed(xml_data)
        # print("Batch:")
        # print(batch)
        if not batch:
            # No more results returned by arXiv
            break

        for paper in batch:
            arxiv_id = paper["id"].split("/")[-1]
            pdf_fname = f"{arxiv_id}.pdf"
            src_fname = f"{arxiv_id}.tar.gz"
            meta_fname = f"{arxiv_id}.txt"

            # (1) Download PDF if link exists
            if paper.get("pdf_link"):
                pdf_path = os.path.join(DOWNLOAD_DIR, pdf_fname)
                try:
                    download_pdf(paper["pdf_link"], pdf_path)
                    print(f"▼ Downloaded PDF: {pdf_fname}")
                except Exception as e:
                    print(f"Failed to download PDF ({arxiv_id}): {e}")

            # (2) Download LaTeX source
            src_path = os.path.join(DOWNLOAD_DIR, src_fname)
            try:
                download_source(arxiv_id, src_path)
                print(f"▼ Downloaded Source: {src_fname}")
            except Exception as e:
                print(f"Failed to download source ({arxiv_id}): {e}")

            # (3) Write metadata .txt
            meta_path = os.path.join(DOWNLOAD_DIR, meta_fname)
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"ID: {paper['id']}\n")
                f.write(f"Title: {paper['title']}\n")
                f.write(f"Authors: {', '.join(paper['authors'])}\n")
                f.write(f"Published: {paper['published']}\n")
                f.write(f"Summary: {paper['summary']}\n")
                f.write(f"PDF Link: {paper.get('pdf_link', 'N/A')}\n")
                f.write(f"Source Link: https://arxiv.org/e-print/{arxiv_id}\n")
            print(f"✔ Saved metadata: {meta_fname}")

            downloaded += 1
            if downloaded >= max_results:
                break

        start += batch_size
        time.sleep(PAUSE_BETWEEN_CALLS)

        extract_tar_gz_files(DOWNLOAD_DIR)

    print(f"\nCrawling complete. Total papers processed: {downloaded}")

# ─── ENTRY POINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: fetch up to 100 “cs.AI” papers (including source) submitted between Jan 1 2025 and May 31 2025
    # crawl_arxiv_with_source(search_query, max_results=10)
    extract_tar_gz_files(directory=DOWNLOAD_DIR)
