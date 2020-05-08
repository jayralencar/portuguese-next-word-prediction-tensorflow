import json
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras.models import Sequential, load_model

with open('data/vocab.json') as f:
    tokenizer = tokenizer_from_json(json.load(f))

model = load_model('saved_models/keras_model_2.h5')
text = " predominante da vasta"
text = " ".join(text.split(" ")[:3])
encoded = tokenizer.texts_to_sequences([text])[0]
print(encoded)

text = " Hoje"
while True:
    parts = text.split(' ')
    if len(parts) < 3:
        encoded = [0, 0, 0][:3-len(parts)] + \
            tokenizer.texts_to_sequences([text])[0]
    else:
        encoded = tokenizer.texts_to_sequences(
            [" ".join(text.split(" ")[-2:])])[0]
    encoded = np.array([encoded])
    encoded = np.reshape(encoded, (encoded.shape[0], encoded.shape[1], 1))
    next = model.predict(encoded, verbose=0)
    for x in next:
        sort = x.argsort()[-3:][::-1]
        for id_ in sort:
            for word, index in tokenizer.word_index.items():
                if index == id_:
                    print(word)

    new_text = input(text+" ->")
    text += " "+new_text
