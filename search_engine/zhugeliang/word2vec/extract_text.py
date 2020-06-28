from zhugeliang.utils.pdf_extractor import pdf_extractor
from zhugeliang.utils.text_process import process_text
from zhugeliang.utils.config import get_book_data_dir, get_data_dir
import glob, os
#import queue
#import threading
#from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, JoinableQueue


def read_file_path(file_path_queue=None, dir_path=None):
    for file_path in glob.glob(dir_path, recursive=True):
        file_path_queue.put(file_path, block=True)

    file_path_queue.join()


def process_pdf(file_path_queue=None, text_queue=None):
    while True:
        try:
            pdf_file_path = file_path_queue.get(block=True)
            print("Process: ", pdf_file_path)
            texts = []
            for text in process_text(pdf_extractor(pdf_file_path, max_page_number=10)):
                texts.append(text)

            text_queue.put("\n".join(texts))

            file_path_queue.task_done()
        except:
            print("Wrong", pdf_file_path)

    text_queue.join()


def write_text(text_queue=None, output_path=None):
    fw = open(output_path, 'w', encoding="utf-8")

    while True:
        text = text_queue.get()
        if text is not None:
            fw.write(text + "\n")

    fw.close()
    text_queue.task_done()
    print("Write done!")


if __name__ == "__main__":
    book_data_dir = os.path.join(get_book_data_dir(), "**/*.pdf")
    text_path = os.path.join(get_data_dir(), "book_text.txt")

    file_path_queue = JoinableQueue(maxsize=10)
    text_queue = JoinableQueue(maxsize=10)

    read_file_path_p = Process(target=read_file_path,
                               kwargs={"file_path_queue": file_path_queue,
                                        "dir_path": book_data_dir})

    read_file_path_p.start()

    process_file_p = [Process(target=process_pdf,
                              kwargs={"text_queue": text_queue,
                                      "file_path_queue": file_path_queue})]

    for p in process_file_p:
        p.daemon = True
        p.start()

    read_file_path_p.join()

    write_text_p = Process(target=write_text,
                           kwargs={"text_queue": text_queue,
                                   "output_path": text_path})

