from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import re

class PreprocessingHelper():

    def __init__(self, root_dir, verbose=True):
        self.root_dir = root_dir
        self.data_directory = root_dir + "/data/"
        self.verbose = verbose
        self.tokenizer = Tokenizer(oov_token="<UNKNOWN>")

    def tokenize_column(self, column_name):
        train_raw, test_raw = self.load_raw_data()
        doc_list = train_raw[column_name].to_list()
        self.clean_doc(doc_list)
        print("doc list cleaned")
        # print(doc_list)
        self.tokenizer.fit_on_texts(doc_list)
        print(self.tokenizer.document_count)
        print(len(train_raw))

    def clean_doc(self, doc_list):
        for idx, doc_text in enumerate(doc_list):
            doc_text = re.sub(r'<br />', "", doc_text)
            doc_text = re.sub(r'[^A-Za-z]+', "", doc_text)
            doc_text = re.sub(r'   ', " ", doc_text)
            doc_text = re.sub(r'  ', " ", doc_text)
            doc_list[idx] = doc_text

    def load_raw_data(self):
        train_raw = pd.read_csv(self.data_directory + "labeledTrainData.tsv", delimiter="\t")
        test_raw = pd.read_csv(self.data_directory + "testData.tsv", delimiter="\t")
        return train_raw, test_raw


