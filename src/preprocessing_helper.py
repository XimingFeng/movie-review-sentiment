from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle


class PreprocessingHelper():

    def __init__(self, root_dir, max_doc_len, vocab_size, verbose=True):
        self.root_dir = root_dir
        self.data_directory = root_dir + "/data/"
        self.max_doc_len = max_doc_len
        self.vocab_size = vocab_size
        self.verbose = verbose

        if os.path.exists(self.data_directory + "preprocessed_data.pickle"):
            with open(self.data_directory + "preprocessed_data.pickle", "rb") as pickle_file:
                self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.tokenizer = pickle.load(pickle_file)
                if verbose:
                    print("preprocessed data loaded from " + self.data_directory + "preprocessed_data.pickle")
        else:
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.tokenizer = self.setup_raw_data()

        if self.verbose:
            print("Training data: {}".format(self.X_train.shape))
            print("Training label: {}".format(self.y_train.shape))
            print("Validation data: {}".format(self.X_val.shape))
            print("Validation lable: {}".format(self.y_val.shape))
            print("Test set {}".format(self.X_test.shape))


    def setup_raw_data(self):
        tokenizer = Tokenizer(oov_token="<UNKNOWN>", num_words=self.vocab_size)
        train_raw, test_raw = self.load_raw_data()
        train_raw_list = train_raw["review"].to_list()
        test_raw_list = test_raw["review"].to_list()
        train_cleaned = self.clean_docs(train_raw_list)
        test_cleaned = self.clean_docs(test_raw_list)
        if self.verbose:
            print("raw text cleaned")
        tokenizer.fit_on_texts(train_cleaned)
        if self.verbose:
            print("dict built from training")
        train_ids = tokenizer.texts_to_sequences(train_cleaned)
        test_ids = tokenizer.texts_to_sequences(test_cleaned)
        X_train = pad_sequences(train_ids, maxlen=self.max_doc_len, padding="post")
        X_test = pad_sequences(test_ids, maxlen=self.max_doc_len, padding="post")
        y_train = np.array(train_raw["sentiment"].to_list(), dtype=np.int32)
        X_train, X_val, y_train, y_val = \
            train_test_split(X_train, y_train, test_size=0.2, random_state=66)

        with open(self.data_directory + "preprocessed_data.pickle", "wb") as pickle_file:
            pickle.dump([X_train, y_train, X_val, y_val, X_test, tokenizer], pickle_file)
        print("preprocessed data dumped into pickle file!")
        return X_train, y_train, X_val, y_val, X_test, tokenizer

    def clean_docs(self, doc_list_raw):
        doc_list_cleaned = doc_list_raw.copy()
        for idx, doc_text in enumerate(doc_list_raw):
            doc_text = doc_text.lower()
            doc_text = re.sub(r'<br />', " ", doc_text)
            doc_text = re.sub(r'[^a-z]+', " ", doc_text)
            doc_text = re.sub(r"   ", " ", doc_text)
            doc_text = re.sub(r"  ", " ", doc_text)
            doc_list_cleaned[idx] = doc_text
        return doc_list_cleaned

    def load_raw_data(self):
        train_raw = pd.read_csv(self.data_directory + "labeledTrainData.tsv", delimiter="\t")
        test_raw = pd.read_csv(self.data_directory + "testData.tsv", delimiter="\t")
        if self.verbose:
            print("data loaded from file")
        return train_raw, test_raw

    def train_batch_generator(self, batch_size):
        start_idx = 0
        while True:
            end_idx = start_idx + batch_size
            if end_idx > len(self.X_train):
                start_idx = 0
                end_idx = batch_size
            yield self.X_train[start_idx: end_idx, :], self.y_train[start_idx: end_idx]
            start_idx += batch_size

    def validation_batch_generator(self, batch_size):
        start_idx = 0
        while True:
            end_idx = start_idx + batch_size
            if end_idx > len(self.X_val):
                start_idx = 0
                end_idx = batch_size
            yield self.X_val[start_idx: end_idx, :], self.y_val[start_idx: end_idx]
            start_idx += batch_size
