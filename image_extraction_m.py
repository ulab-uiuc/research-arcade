import fitz
from pathlib import Path

def extract_with_pymupdf(src_pdf: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    doc = fitz.open(str(src_pdf))
    for i, page in enumerate(doc, start=1):
        # extract the Form XObjects as raw PDF streams
        for name, pdf_bytes in page.get_xobject_pdf().items():
            out = out_dir / f"page{i}_{name}.pdf"
            out.write_bytes(pdf_bytes)
            print("Saved vector PDF object:", out)

if __name__ == "__main__":
    extract_with_pymupdf(Path("pdf/2506.20807v1.pdf"), Path("extracted_objs"))
