import pikepdf
from pathlib import Path

def wrap_form_xobject(src_pdf_path: Path, page_num: int, xobj_name: str, out_pdf_path: Path):
    # 1) Open source and grab the XObject stream
    src = pikepdf.Pdf.open(str(src_pdf_path))
    page = src.pages[page_num - 1]
    print(page.Resources)
    xobjs = page.Resources.get("/XObject", {})
    if xobj_name not in xobjs:
        raise KeyError(f"{xobj_name} not found on page {page_num}")
    form = xobjs[xobj_name]

    # 2) Create a new PDF
    new_pdf = pikepdf.Pdf.new()

    # 3) Copy the form (and its dependencies) into new_pdf
    form_copy = new_pdf.copy_foreign(form)

    # 4) Clone the page (to preserve MediaBox, etc.), then overwrite its Resources/Contents
    new_pdf.pages.append(page)
    new_page = new_pdf.pages[-1]

    new_page.Resources = pikepdf.Dictionary({
        "/XObject": pikepdf.Dictionary({ xobj_name: form_copy })
    })

    draw_cmd = b"q\n1 0 0 1 0 0 cm " + xobj_name.encode("utf-8") + b" Do\nQ"
    content_stream = pikepdf.Stream(new_pdf, draw_cmd)
    new_page.Contents = new_pdf.make_indirect(content_stream)

    # 5) Save
    new_pdf.save(str(out_pdf_path))
    print("Wrapped PDF written to", out_pdf_path)

if __name__ == "__main__":
    src_pdf     = Path("pdf/2506.20807v1.pdf")
    wrapped_pdf = Path("figure_im1_wrapped.pdf")

    wrap_form_xobject(
        src_pdf_path=src_pdf,
        page_num=1,
        xobj_name="/Im1",           # the resource name you saw
        out_pdf_path=wrapped_pdf
    )
