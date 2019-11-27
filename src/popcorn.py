import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim import corpora

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

    def build_dictionary(self, sentence_list):
        return corpora.Dictionary(sentence_list)

    

