import json
import numpy as np
from keras.models import load_model
import heapq
from nltk.tokenize import RegexpTokenizer

SEQUENCE_LENGTH = 3
SENTENCE_LENGTH = 10
tokenizer = RegexpTokenizer(r'\w+')

with open('data/vocab.json') as f:
    unique_words = json.load(f)

model = load_model('saved_models/word_prediction.h5')

def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(unique_words)))
    for t, word in enumerate(text.split()):
        x[0, t, unique_words.index(word)] = 1
    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completions(text, n=3):
    if text == "":
        return("0")
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [unique_words[idx] for idx in next_indices]

sentence = 'os únicos envolvidos'
words = sentence.split(" ")
while len(words) < SENTENCE_LENGTH:
    q = (["0"]*SEQUENCE_LENGTH)[:SEQUENCE_LENGTH-len(words)] + words
    sentence = ' '.join(q)
    seq = " ".join(tokenizer.tokenize(sentence.lower())[0:SEQUENCE_LENGTH])
    preds = predict_completions(seq, 1)
    if len(preds) > 0:
        words.append(preds[0])

print("Generated sentence: ", " ".join(words))
