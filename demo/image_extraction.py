from PyPDF2 import PdfReader

reader = PdfReader("pdf/2506.20807v1.pdf")

pages = reader.pages
count = 0

for page in pages:
    for image_file_object in page.images:
        with open(str(count) + image_file_object.name, "wb") as fp:
            fp.write(image_file_object.data)
            count += 1

