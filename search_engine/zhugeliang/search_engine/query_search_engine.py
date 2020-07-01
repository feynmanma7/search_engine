from zhugeliang.search_engine.seach_engine import SearchEngine
from zhugeliang.utils.config import get_model_dir
import sys, os

if __name__ == '__main__':
    model_dir_path = os.path.join(get_model_dir(), "v1")

    se = SearchEngine()
    se.load_model(model_dir_path=model_dir_path)
    print("Load model done!")

    id2doc_dict = {doc_id: doc_path for doc_path, doc_id in se.doc2id_dict.items()}

    query = "word2vec milkov"
    doc_id_list = se.search(query=query)

    if doc_id_list is None:
        sys.exit(0)

    print("\n\n\n")
    print("Total number of files of query \"%s\" is: %d\n" % (query, len(doc_id_list)))

    doc_path_list = [id2doc_dict[doc_id] for doc_id in doc_id_list]
    for i, doc_path in enumerate(doc_path_list):
        print("%d.\t%s" % (i+1, '/'.join(doc_path.split('/')[4:])))

