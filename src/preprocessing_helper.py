from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle


class PreProcessingHelper:
    """
    This class is used to handle data pre-processing
    """

    def setup_tokenizer(self, cleaned_corpus, vocab_size, save_tokenizer=False, file_path=None):
        """
        Construct tokenizer based on cleaned corpus

        :param list cleaned_corpus: a list of documents (str). see function clean_docs for cleaning method
        :param int vocab_size: the size of vocabulary
        :param bool save_tokenizer: if tokenizer is going to be saved in a pickle file
        :param str file_path: specify the file path where tokenizer is going to be saved
        :return: constructed tokenizer
        """
        tokenizer = Tokenizer(oov_token="<UNKNOWN>", num_words=vocab_size)
        tokenizer.fit_on_texts(cleaned_corpus)
        if save_tokenizer:
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(tokenizer, pickle_file)
        return tokenizer

    def get_tokenizer(self, file_path):
        """
        Get tokenizer from pickle file

        :param file_path: path to pickle file
        :return: tokenizer, None if there is no file saved
        """
        tokenizer = None
        if os.path.exists(file_path):
            with open(file_path, 'rb') as pickle_file:
                tokenizer = pickle.load(pickle_file)
        return tokenizer

    def clean_docs(self, doc_list_raw):
        """
        Clean raw corpus, clean up items include:

        1. lower case all words
        2. allow only alphabetical letters
        3. remove extra space

        :param doc_list_raw: list raw documents (str)
        :return: cleaned corpus
        """
        doc_list_cleaned = doc_list_raw.copy()
        for idx, doc_text in enumerate(doc_list_raw):
            doc_text = doc_text.lower()
            doc_text = re.sub(r'<br />', " ", doc_text)
            doc_text = re.sub(r'[^a-z]+', " ", doc_text)
            doc_text = re.sub(r"   ", " ", doc_text)
            doc_text = re.sub(r"  ", " ", doc_text)
            doc_list_cleaned[idx] = doc_text
        return doc_list_cleaned

    def clean_corpus_to_ids(self, clean_corpus, tokenizer, max_len):
        """
        convert cleaned corpus to word ids

        :param list(str) clean_corpus: cleaned corpus
        :param tokenizer: tokenizer
        :param int max_len: max length of word for each documents
        :return: numpy array with shape (len(clean_corpus), max_len)
        """
        ids = tokenizer.texts_to_sequences(clean_corpus)
        ids_pad = pad_sequences(ids, max_len, padding="post")
        return ids_pad

    def save_train_val_test(self, data_list, file_path):
        """
        save preprocessed data into a pickle file

        :param data_list: [X_train, y_train, X_val, y_val, X_test]
        :param file_path: path to the pickle file
        """

        with open(file_path, "wb") as pickle_file:
            pickle.dump(data_list, pickle_file)
        print("preprocessed data dumped into pickle file!")

    def get_train_val_test(self, file_path):
        """
        Get preprocessed data from pickle file

        :param str file_path: path to pickle file
        :return list: list of preprocessed data [X_train, y_train, X_val, y_val, X_test]
        """

        preprocessed_data = None
        if os.path.exists(file_path):
            with open(file_path, "rb") as pickle_file:
                preprocessed_data = pickle.load(pickle_file)
        return preprocessed_data

    def train_batch_generator(self, batch_size):
        """
        Generator for training batch, once it reach the end of training set, start over again

        :param int batch_size: size of batch
        :return: training batch (yield)
        """

        start_idx = 0
        while True:
            end_idx = start_idx + batch_size
            if end_idx > len(self.X_train):
                start_idx = 0
                end_idx = batch_size
            yield self.X_train[start_idx: end_idx, :], self.y_train[start_idx: end_idx]
            start_idx += batch_size

    def validation_batch_generator(self, batch_size):
        """
        Generator for validation batch, once it reach the end of validation set, start over again

        :param int batch_size: size of batch
        :return: validation batch (yield)
        """

        start_idx = 0
        while True:
            end_idx = start_idx + batch_size
            if end_idx > len(self.X_val):
                start_idx = 0
                end_idx = batch_size
            yield self.X_val[start_idx: end_idx, :], self.y_val[start_idx: end_idx]
            start_idx += batch_size
