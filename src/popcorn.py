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
    
    def read_tsv_file(self, path):
        return pd.read_csv(path, delimiter="\t")
    
    def load_raw_data(self, folder_path):
        train_data = self.read_tsv_file(folder_path + "labeledTrainData.tsv")
        test_data = self.read_tsv_file(folder_path + "testData.tsv")
        return train_data, test_data


    def clean_word(self, word):
        poter_stemmer = PorterStemmer()
        word = re.sub(r"<br />", " ", word)
        word = re.sub(r"[^a-z]", " ", word)
        word = re.sub(r"   ", " ", word)  # Remove any extra spaces
        word = re.sub(r"  ", " ", word)
        word = poter_stemmer.stem(word)
        return word

    def clean_text(self, text, remove_stopwords=True):
        text_list = word_tokenize(text)
        if remove_stopwords:
            stop_wds = set(stopwords.words("english"))
            text_list = [self.clean_word(w) for w in text_list if not w in stop_wds]

        text = " ".join(text_list)
        return text

