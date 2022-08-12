import numpy
import tensorflow
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
# from keras.layers.embeddings  import Embedding
from keras.preprocessing import sequence


from keras.layers import Dense,LSTM,Embedding

import numpy as np
import jieba

from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from keras import regularizers
from keras import layers
from keras import losses
from keras import preprocessing
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
top_words = 7000
dev_set = pd.read_csv("./dev.tsv",sep='\t',names=['text','category','id'])
train_set = pd.read_csv("./train.tsv",sep='\t',names=['text','category','id'])
test_set= pd.read_csv("./test.tsv",sep='\t',names=['text','category','id'])
train_set['text'] = train_set.text.apply(lambda x: " ".join(jieba.cut(x)))
dev_set['text']= dev_set.text.apply(lambda x: " ".join(jieba.cut(x)))
test_set['text']= test_set.text.apply(lambda x: " ".join(jieba.cut(x)))

num_words = 20000
tokenizer = Tokenizer(num_words=num_words,oov_token="unk")
tokenizer.fit_on_texts(train_set['text'].tolist())


X_train,  X_test, y_train, y_test= train_test_split(train_set['text'].tolist(),\
                                                      train_set['category'].tolist(),\
                                                      test_size=0.1,\
                                                      stratify = train_set['category'].tolist(),\
                                                      random_state=0)

X_train = np.array(tokenizer.texts_to_sequences(X_train))
X_test = np.array(tokenizer.texts_to_sequences(X_test))
y_train = numpy.array(y_train)
y_test = numpy.array(y_test)


# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

print(X_train[0])
print(y_train[0])
print('Shape of training data: ')
print(X_train.shape)
print(y_train.shape)
print('Shape of test data: ')
print(X_test.shape)
print(y_test.shape)

max_words = 450
X_train =pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)
# Building the CNN Model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(5, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Fitting the data onto model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Getting score metrics from our model
scores = model.evaluate(X_test, y_test, verbose=0)
# Displays the accuracy of correct sentiment prediction over test data
print("Accuracy: %.2f%%" % (scores[1]*100))