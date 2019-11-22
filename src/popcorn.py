import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re

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
    
    def clean_text(self, text, remove_stopwords=True):
        text_list = text.lower().split()
        if remove_stopwords:
            stop_wds = set(stopwords.words("english"))
            text_list = [w for w in text_list]
        text = " ".join(text_list)
        text = re.sub(r"<br />", " ", text)
        text = re.sub(r"[^a-z]", " ", text)
        text = re.sub(r"   ", " ", text) # Remove any extra spaces
        text = re.sub(r"  ", " ", text)
        return text
    