import numpy as np
from numpy import array
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop

data = ""
with open("./data/corpus.txt") as f:
    data = f.read().strip()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# create line-based sequences
sequences = list()
print("TOKENS: ", len(tokenizer.word_index.items()))
for encoded in tokenizer.texts_to_sequences(data.split("\n")):
    if len(encoded) > 0:
        for i in range(0, len(encoded) - 2):
            sequences.append(encoded[i:i+3])
# print(sequences)
print("TOKENS: ", len(tokenizer.word_index.items()))
print('Total Sequences: %d' % len(sequences))
sequences = np.array(sequences)
X, y = sequences[:,:-1], to_categorical(sequences[:,-1], num_classes=vocab_size)
# print(X.shape)
# print(X)
X =  np.reshape(X,(X.shape[0], X.shape[1],1))
# print(X.shape)
# # print(X[0])
# print(vocab_size)
# # define model
# # i = tf.keras.layers.Input(shape=(X.shape[1],))
# # e = tf.keras.layers.Embedding(vocab_size, 10, input_length=2)(i)
# # l = tf.keras.layers.LSTM(10)(e)
# # d = tf.keras.layers.Dense(vocab_size, activation='softmax')(l)
# # model = tf.keras.Model(inputs=i, outputs=[d])

model = Sequential()
model.add(LSTM(256, input_shape=(2,1),return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

print(model.summary())
# # compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # tf.keras.utils.plot_model(model)
# print(X.shape)
print(y.shape)
# print(y[0])
model.fit(X, y, epochs=10)

# # model.save('keras_model_2.h5')

text = " observa o novo"
text = " ".join(text.split(" ")[:3])
print(text)
encoded = tokenizer.texts_to_sequences([text])[0]
print("TOKENS: ", len(tokenizer.word_index.items()))
# print(encoded)
encoded = array([encoded])
encoded = np.reshape(encoded, (encoded.shape[0], encoded.shape[1],1))
# print(encoded.shape)
next = model.predict(encoded, verbose=0)
print(len(next))
for x in next:
    print(len(x))
    print(x[0])
    sort = x.argsort()[-3:][::-1]
    # next_word_token = np.argmax(x)    # map predicted word index to word
    # print(sort)
    for id_ in sort:
        # print(x)
        for word, index in tokenizer.word_index.items():
            if index == id_:
                print(word + " ")

