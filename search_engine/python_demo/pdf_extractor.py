import pdfplumber
import traceback


def pdf_extractor(file_path=None, max_page_number=None):
    with pdfplumber.open(file_path) as f:

        if len(f.pages) > max_page_number:
            return None

        for page in f.pages:
            text = page.extract_text()
            yield text



if __name__ == '__main__':
    file_path = 'data/Attention Is All You Need.pdf'
    gen = pdf_extractor(file_path=file_path)
    for text in gen:
        print(text)