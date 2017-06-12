# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tf_data import TF_Data
import pandas
from keras.models import load_model
from os.path import isfile
import os
from keras.optimizers import Adam
import pickle
def train(filename, model_name, day='tomorrow'):
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint(model_name, monitor='val_acc', mode='auto', verbose=1, save_best_only=True)
    earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1)
    df = pandas.read_csv(filename, sep='\t')
    headlines = df['normalized_headline'].as_matrix()
    top_words = 2000

    data = TF_Data(filename, top_words=top_words)
    pickle.dump(data, open(filename.replace(".csv", ".p"), "wb"))
    # load the dataset but only keep the top n words, zero the rest
    (X_train, y_train), (X_test, y_test) = data.load_data(day=day)
    print X_train.shape

    # truncate and pad input sequences
    max_review_length = 100
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if isfile(model_name) and False:
        print('Checkpoint Loaded')
        model = load_model(model_name)
    print(model.summary())

    model.fit(X_train, y_train, validation_split=0.02, epochs=100, batch_size=64, callbacks=[checkpoint, earlyStopping, reduceLR])
    # Final evaluation of the model
    model = load_model(model_name)
    scores = model.evaluate(X_test, y_test, verbose=0)
    fd = open('accuracy.csv','a')
    CsvRow = [filename, day, "Accuracy: %.2f%%" % (scores[1]*100)]
    print CsvRow
    fd.write(", ".join(CsvRow) + "\n")
    fd.close()

for f in os.listdir("../data/all_data"):
    if(f.endswith('.csv')):
        for d in ['today', 'tomorrow', 'day_after_tomorrow']:
            train(os.path.join("../data/all_data", f), f.replace(".csv","_" + d +"_.hdf5"), d)

