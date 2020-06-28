import pdfplumber


def pdf_extractor(file_path=None, max_page_number=None):
    with pdfplumber.open(file_path) as f:
        for i, page in enumerate(f.pages):

            if max_page_number is not None and i == max_page_number:
                break

            text = page.extract_text()

            yield text