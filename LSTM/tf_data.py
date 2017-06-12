import pandas
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from unidecode import unidecode
from keras.utils.np_utils import to_categorical
import numpy as np
class TF_Data:
    def __init__(self, data_file, validation_split=0.1, top_words=5000):
        self.top_words = top_words
        self.validation_split = validation_split
        self.df = pandas.read_csv(data_file, sep='\t')

        self.headlines = self.df['normalized_headline'].as_matrix()
        self.tokenizer = Tokenizer(top_words)
        self.tokenizer.fit_on_texts(self.headlines)
        self.all_x = self.tokenizer.texts_to_sequences(self.headlines)



    def load_data(self, day='tomorrow'):
        np.random.seed(0)
        self.all_y = self.df[day].as_matrix()
        idx = np.arange(len(self.all_x))
        np.random.shuffle(idx)
        self.all_x = np.array(self.all_x)[idx]
        self.all_y = np.array(self.all_y)[idx]
        split = int(self.validation_split * len(self.all_x))
        training_x = self.all_x[split:]
        training_y = self.all_y[split:]
        validation_x = self.all_x[:split]        
        validation_y = self.all_y[:split]
        np.random.seed(None)
        return (training_x, training_y), (validation_x, validation_y)

    def test_sentence(self, text):
        return np.array(sequence.pad_sequences(self.tokenizer.texts_to_sequences([unidecode(text)]),maxlen=100))
        


