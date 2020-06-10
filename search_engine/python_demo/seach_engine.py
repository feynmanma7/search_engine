from python_demo.store import load_model, save_model
from python_demo.pdf_extractor import pdf_extractor
from python_demo.text_process import process_english_text
from python_demo.utils import last_time
import glob
import os


class SearchEngine:
    def __init__(self):
        super(SearchEngine, self).__init__()

        self.info = 'Local search engine'

        self.doc_id = 0
        self.doc2id_dict = {}
        self.id2doc_dict = {}
        self.inverted_index = {}

    def load_model(self, model_dir_path=None):
        self.model = load_model(model_dir_path=model_dir_path)

        if 'doc_id' in self.model:
            self.doc_id = self.model['doc_id']

        if 'doc2id_dict' in self.model:
            self.doc2id_dict = self.model['doc2id_dict']

        if 'id2doc_dict' in self.model:
            self.id2doc_dict = self.model['id2doc_dict']

        if 'inverted_index' in self.model:
            self.inverted_index = self.model['inverted_index']


    def search(self, query=None):
        ans = None

        for word in query.split(' '):
            word = word.lower()
            if word in self.inverted_index:
                doc_id = self.inverted_index[word]
                if ans is None:
                    ans = doc_id
                else:
                    ans &= doc_id

        return ans

    def _process_text(self, text_generator=None):
        for text in text_generator:
            for word in process_english_text(text=text):
                # TODO
                # word: doc_id should remove duplicate
                if word not in self.inverted_index:
                    self.inverted_index[word] = set([self.doc_id])
                else:
                    self.inverted_index[word] &= set([self.doc_id])

    @last_time
    def _build_by_doc(self, file_path=None, doc_type=None):
        if doc_type == 'pdf':
            text_generator = pdf_extractor(file_path=file_path)
            self._process_text(text_generator=text_generator)

    def _process_file_path(self, file_path=None):
        is_processed = False
        if file_path in self.doc2id_dict:
            is_processed = True
            return is_processed

        self.doc_id += 1
        self.doc2id_dict[file_path] = self.doc_id
        self.id2doc_dict[self.doc_id] = file_path

        return is_processed


    def _build_by_dir(self, dir_path=None, doc_type=None):
        for i, file_path in enumerate(glob.glob(os.path.join(dir_path, '*' + doc_type), recursive=True)):
            is_processed = self._process_file_path(file_path=file_path)
            if is_processed:
                continue
            print(i+1, file_path)

            self._build_by_doc(file_path=file_path, doc_type=doc_type)


    def _sort_doc_id(self):
        for word in self.inverted_index.keys():
            self.inverted_index[word] = set(sorted(self.inverted_index[word]))


    def build_inverted_index(self, dir_path=None, model_dir_path=None):
        # inverted_index: word: sorted(doc_id, doc_id, ..., doc_id)
        self.load_model(model_dir_path=model_dir_path)

        self._build_by_dir(dir_path=dir_path, doc_type='pdf')

        self._sort_doc_id()

        self.model['doc_id'] = self.doc_id
        self.model['doc2id_dict'] = self.doc2id_dict
        self.model['id2doc_dict'] = self.id2doc_dict
        self.model['inverted_index'] = self.inverted_index

        save_model(model_dir_path=model_dir_path, model=self.model)

