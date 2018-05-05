import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Conv1D, MaxPooling1D, Dropout, Activation
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from collections import Counter
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def load_vocab():
    # load the vocabulary
    vocab_filename = 'vocab4.txt'
    vocab2 = load_doc(vocab_filename)
    vocab2 = vocab2.split()
    return set(vocab2)

dictionary = Counter()
def load_dictionary():
    with open("dictionary.txt", "r") as f:
        for l in f:
            tokens = l.split()
            dictionary.update({tokens[0] : int(tokens[1])})

if __name__ ==  '__main__':
    # print("Reading Data")
    # data = pd.read_csv("text_data.csv")
    # load_dictionary()
    # print(dictionary.most_common(50))

    # # keep tokens with a min occurrence
    # min_occurance = 10
    # vocab = [k for k,c in dictionary.items() if c >= min_occurance]
    # print('Number of words to query reduced to: ' + str(len(vocab)))

    # print("Spawning Processes")
    # n_jobs = 12
    # inputs = np.array_split(data, n_jobs)
    # tasks = []
    # j = 0
    # out = mp.Queue()
    # widgets = mp.Queue()
    # processes = []
    # results = []

    # from Filter import filter
    # for i in inputs:
    #     processes.append(mp.Process(target=filter, args=(i,vocab,out,widgets)))

    # for p in processes:
    #     p.start()
    # bars = [widgets.get() for p in processes]
    # for f in bars:
    #     print(f)
        
    # results = [out.get() for p in processes]
    # print("Joining Processes")
    # for p in processes:
    #     p.join()
    # frames = []
    # for result in results:
    #     frames.append(result)
        
    # data = pd.concat(frames) 

    # data.to_csv("filtered_data.csv")

    data = pd.read_csv("filtered_data.csv")
    data = data.dropna(axis=0, how='any')
    print(data.head())

    y = [0 if rating <= 3 else 1 for rating in data['stars']]

    print("Splitting")

    x_train, x_test, y_train, y_test = train_test_split(data['text'], y, test_size=0.20, random_state=17)
    #x_test, x_rest, y_test, y_rest = train_test_split(x_rest, y_rest, test_size=0.80, random_state=17)
    print("Length of Train set: " + str(len(x_train)))
    print("Length of Test set: " + str(len(x_test)))

    max_features = 100000

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(x_train.values)

    print("Tokenizing")
    sequences = tokenizer.texts_to_sequences(x_train.values)
    X = pad_sequences(sequences)
    y = y_train

    #print(X)

    #print(y)
    # define network
    es = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    print(model.summary())
    history = model.fit(X, y, validation_split=0.20, batch_size=1024, epochs=3, verbose=1, callbacks=[es])
  
    # evaluate
    loss, acc = model.evaluate(pad_sequences(tokenizer.texts_to_sequences(x_test.values)), y_test, verbose=1)
    print('Test Accuracy: %f' % (acc*100))
    print('Test Loss: %f' % (loss))
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.figure(2)
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save('lstm.h5')