import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class PopcornHelper():
    
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.x_test = None
        self.stop_wds = {
            r"<br />", r"[^a-z]",
        }
    
    def read_tsv_file(self, path):
        return pd.read_csv(path, delimiter="\t")
    
    def load_raw_data(self, folder_path):
        train_data = self.read_tsv_file(folder_path + "labeledTrainData.tsv")
        test_data = self.read_tsv_file(folder_path + "testData.tsv")
        return train_data, test_data

    def is_valid_word(self, word, remove_stop_wds):
        is_valid = True
        stop_wds = set(stopwords.words("english"))
        if (word in stop_wds and remove_stop_wds) or \
                re.match(r"<br />", word)  or \
                re.match(r"[^a-z]", word) or \
                re.match(r"   ", word):
            is_valid = False
        return is_valid

    def clean_text(self, text, remove_stopwords=True):
        text_list = word_tokenize(text)
        poter_stemmer = PorterStemmer()
        text_list = [poter_stemmer.stem(w.lower()) for w in text_list
                     if self.is_valid_word(w.lower(), remove_stopwords)]
        text = " ".join(text_list)
        return text

