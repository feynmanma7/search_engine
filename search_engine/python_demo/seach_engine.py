from python_demo.store import load_model as _load_model, save_model as _save_model
from python_demo.pdf_extractor import pdf_extractor
from python_demo.text_process import process_english_text
from python_demo.utils import last_time, intersect_sorted_list
import glob
import os


class SearchEngine:
    def __init__(self, max_page_number=None):
        super(SearchEngine, self).__init__()

        self.info = 'Local search engine'
        self.max_page_number = max_page_number

        self.doc_id = 0
        self.doc2id_dict = {}
        self.inverted_index = {}

    def load_model(self, model_dir_path=None):
        self.model = _load_model(model_dir_path=model_dir_path)

        if 'doc2id_dict' in self.model:
            self.doc2id_dict = self.model['doc2id_dict']

        if 'inverted_index' in self.model:
            self.inverted_index = self.model['inverted_index']

        self.doc_id = len(self.doc2id_dict)

    def save_model(self, model_dir_path=None):
        self._sort_doc_id()

        self.model['doc2id_dict'] = self.doc2id_dict
        self.model['inverted_index'] = self.inverted_index

        _save_model(model_dir_path=model_dir_path, model=self.model)

    def search(self, query=None):
        ans = None

        for word in query.split(' '):
            word = word.lower()
            if word in self.inverted_index:
                doc_id_list = self.inverted_index[word]
                if ans is None:
                    ans = doc_id_list
                else:
                    ans = intersect_sorted_list(ans, doc_id_list)

        return ans

    def _process_text(self, text_generator=None):
        # Make sure a word only exists once in one document.
        word_dict = {}

        for text in text_generator:
            if text is None or len(text) == 0:
                continue

            for word in process_english_text(text=text):

                if word in word_dict:
                    continue
                word_dict[word] = True

                if word not in self.inverted_index:
                    self.inverted_index[word] = [self.doc_id]
                else:
                    self.inverted_index[word].append(self.doc_id)


    @last_time
    def _build_by_doc(self, file_path=None, doc_type=None):
        if doc_type == 'pdf':
            text_generator = pdf_extractor(file_path=file_path, max_page_number=self.max_page_number)
            self._process_text(text_generator=text_generator)

    def _build_by_dir(self, dir_path=None, model_dir_path=None, doc_type=None):
        batch_size = 10
        i = 0
        for file_path in glob.glob(os.path.join(dir_path, '*' + doc_type), recursive=True):
            if file_path in self.doc2id_dict:
                continue
            i += 1

            print(i, file_path)
            self._build_by_doc(file_path=file_path, doc_type=doc_type)
            self.doc2id_dict[file_path] = self.doc_id
            self.doc_id += 1

            if i % batch_size == 0:
                # Flush model to disk
                self.save_model(model_dir_path=model_dir_path)

    def _sort_doc_id(self):
        for word in self.inverted_index.keys():
            self.inverted_index[word] = sorted(self.inverted_index[word])

    def build_inverted_index(self, dir_path=None, model_dir_path=None):
        # inverted_index: word: sorted(doc_id, doc_id, ..., doc_id)
        self.load_model(model_dir_path=model_dir_path)

        try:
            self._build_by_dir(dir_path=dir_path, model_dir_path=model_dir_path, doc_type='pdf')
        except:
            self.save_model(model_dir_path=model_dir_path)

            import traceback
            traceback.print_exc()

        self.save_model(model_dir_path=model_dir_path)

