import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dropout
from keras.layers.core import Dense, Activation, Dropout, RepeatVector

data = ""
with open("./data/pg54829.txt") as f:
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
X =  np.reshape(X,(X.shape[0], X.shape[1],1))

model = Sequential()
model.add(LSTM(256, input_shape=(2,1),return_sequences = False))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

print(model.summary())
# # compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# tf.keras.utils.plot_model(model)

model.fit(X, y, epochs=50)

model.save('saved_models/keras_model_2.h5')

text = " parece estar chorando"
text = " ".join(text.split(" ")[:3])
encoded = tokenizer.texts_to_sequences([text])[0]
encoded = np.array([encoded])
encoded = np.reshape(encoded, (encoded.shape[0], encoded.shape[1],1))
next = model.predict(encoded, verbose=0)
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
                print(text+ " -> "+ word)
                # print(word + " ")

