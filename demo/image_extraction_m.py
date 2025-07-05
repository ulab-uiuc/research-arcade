import subprocess
import shlex
from pathlib import Path
import glob

def extract_xobject_pdfs(master_pdf: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    # run mutool extract from the PDF’s own directory
    subprocess.run(
        ["mutool", "extract", master_pdf.name],
        check=True,
        cwd=str(master_pdf.parent)   # cd into e.g. "./pdf"
    )
    # then move generated objs into out_dir
    for f in master_pdf.parent.glob("obj*.pdf"):
        f.rename(out_dir / f.name)
    return list(out_dir.glob("obj*.pdf"))

def render_xobject_to_png(xobj_pdf: Path, dpi: int = 300):
    """
    Run `mutool draw -r {dpi} -o {stem}.png {xobj_pdf}`
    """
    png_path = xobj_pdf.with_suffix(".png")
    subprocess.run([
        "mutool", "draw",
        "-r", str(dpi),
        "-o", str(png_path),
        str(xobj_pdf)
    ], check=True)
    return png_path

if __name__ == "__main__":
    master = Path("pdf/2506.20807v1.pdf")
    out   = Path("extracted_objs")

    # 1) extract all objects
    xobjs = extract_xobject_pdfs(master, out)
    print("Found embedded PDFs:", xobjs)

    # 2) (optional) render each to PNG
    for pdf in xobjs:
        png = render_xobject_to_png(pdf, dpi=300)
        print("Rendered to:", png)
