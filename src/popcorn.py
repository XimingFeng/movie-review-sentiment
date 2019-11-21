import numpy as np
import pandas as pd
from nltk.corpus import stopwords

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
    
    def clean_text(text, remove_stopwords=True):
        text_list = review_text.lower().split()
        if remove_stopwords:
            stop_wds = set(stopwords.words("English"))
            text_list = [wd for wr in text_list if wd not in stop_wds]
        text = "".join(textlist)
        text = re.sub(r"<br />", " ", text)
        text = re.sub(r"[^a-z]", " ", text)
        text = re.sub(r"   ", " ", text) # Remove any extra spaces
        text = re.sub(r"  ", " ", text)
        return text
    