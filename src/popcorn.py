import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim import corpora
import numpy as np
import pickle
import os


class PopcornHelper():

    def __init__(self, root_directory):
        self.X_train = None
        self.y_train = None
        self.x_test = None
        self.root_directory = root_directory
        self.data_directory = root_directory + "/data/"
    
    def read_tsv_file(self, path):
        return pd.read_csv(path, delimiter="\t")
    
    def load_raw_data(self):
        train_data = self.read_tsv_file(self.data_directory + "labeledTrainData.tsv")
        test_data = self.read_tsv_file(self.data_directory + "testData.tsv")
        return train_data, test_data

    def is_valid_word(self, word, remove_stopwords):
        is_valid = True
        word = word.lower()
        stop_wds = set(stopwords.words("english"))
        if (remove_stopwords and (word in stop_wds)):
            is_valid = False
        elif not re.match(r"^[a-z]+$", word):
            is_valid = False
        return is_valid

    def clean_sentence(self, text, remove_stopwords=True):
        wds_list = word_tokenize(text)
        poter_stemmer = PorterStemmer()
        wds_list = [poter_stemmer.stem(w.lower()) for w in wds_list
                     if self.is_valid_word(w, remove_stopwords)]
        return wds_list

    def clean_dtset(self, dtframe, sentence_col_name, remove_stopwords=True):
        sentences_list = []
        for i in range(len(dtframe)):
            wds_list = self.clean_sentence(dtframe.iloc[i][sentence_col_name], remove_stopwords)
            sentences_list.append(wds_list)
        return sentences_list

    def build_save_dictionary(self, sentence_list):
        corpora_dict = corpora.Dictionary(sentence_list)
        corpora_dict.save(self.data_directory + "corpora_dict.dict")
        return corpora_dict

    def setup_train_test(self):
        train_raw, test_raw = self.load_raw_data()
        train_doc_cleaned = self.clean_dtset(train_raw, sentence_col_name="review")
        test_doc_cleaned = self.clean_dtset(test_raw, sentence_col_name="review")
        corpora_dict = self.build_save_dictionary(train_doc_cleaned)
        X_train = self.doc_2_ids(train_doc_cleaned, corpora_dict)
        X_test = self.doc_2_ids(test_doc_cleaned, corpora_dict)
        y_train = np.array(train_raw.sentiment)
        self.dump_cleaned_data((X_train, X_test, y_train))
        return X_train, X_test, y_train

    def dump_cleaned_data(self, dump_obj):
        dump_file_name = self.data_directory + "cleaned_data.pickle"
        with open(dump_file_name, "wb") as dump_file:
            pickle.dump(dump_obj, dump_file)


    def doc_2_ids(self, sentence_list, corpora_dict):
        doc_ids = []
        for i in range(len(sentence_list)):
            wd_list = corpora_dict.doc2idx(sentence_list[i])
            doc_ids.append(wd_list)
        return doc_ids

    def get_train_test(self):
        if not os.path.exists(self.data_directory + "cleaned_data.pickle"):
            print("Data not cleaned yet. Start to clean up and save cleaned data")
            X_train, X_test, y_train = self.setup_train_test()
        else:
            print("Cleaned data exists, load directly from pickle file")
            with open(self.data_directory + "cleaned_data.pickle", "rb") as dump_file:
                X_train, X_test, y_train = pickle.load(dump_file)
        return X_train, X_test, y_train

    def pad_sequence(self, input_data, max_len):
        sequence_num = len(input_data)
        padded_sequence = np.zeros([sequence_num, max_len])
        for i in range(sequence_num):
            sequence_len = len(input_data[i])
            if sequence_len > max_len:
                padded_sequence[i, :] = input_data[i][: max_len]
            else:
                padded_sequence[i, :sequence_len] = input_data[i]
        return padded_sequence

