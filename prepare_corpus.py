from os import listdir
from os.path import isfile, join
dataset_path = './data/fapesp-corpora/corpora/pt/data'
onlyfiles = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

trainin_text = ''
for file_ in onlyfiles[:10]:
    trainin_text += open(dataset_path+"/"+file_).read()

training_corpus = open('./data/training_corpus.txt', 'w')
training_corpus.write(trainin_text)
