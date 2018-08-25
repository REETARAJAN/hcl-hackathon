from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
print("You have TensorFlow version", tf.__version__)
df = pd.read_csv('train.csv', encoding='latin-1')
df.head()
col = ['comment_text','toxic','severe_toxic','obscene','threat','insult', 'identity_hate']
df = df[col]
df = df[pd.notnull(df['comment_text'])]
df.head()
df.isnull().sum()
print(df['threat'].value_counts())
train_size = int(len(df) * .9)
print("Train size: %d" % train_size)
print("Test size: %d" % (len(df) - train_size))

train_narrative = df['comment_text'][:train_size]
train_product = df['threat'][:train_size]


test_narrative = df['comment_text'][train_size:]
test_product = df['threat'][train_size:]



max_words = 10000

tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_narrative) # only fit on train
x_train = tokenize.texts_to_matrix(train_narrative)
x_test = tokenize.texts_to_matrix(test_narrative)
encoder = LabelEncoder()
encoder.fit(train_product)
y_train = encoder.transform(train_product)
y_test = encoder.transform(test_product)
num_classes = np.max(y_train) + 1


y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
batch_size = 24
epochs = 10
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
text_labels = encoder.classes_

for i in range(50):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    print(test_narrative.iloc[i][:50], "...")
    print('Actual label:')
    print(test_product.iloc[i])
    print("\nPredicted label: ")
    print(predicted_label)