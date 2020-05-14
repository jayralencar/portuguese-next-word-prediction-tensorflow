# portuguese-next-word-prediction-tensorflow
Sugestão de palavras com Tensorflow

Instale as dependências
```
pip install tensorflow
pip install keras
pip install numpy
pip install matplotlib
pip install nltk
```

Baixe e prepare o corpus
```
make corpus
```

Treine o modelo
```
python train.py
```

Teste o modelo
```
python test.py
```

Gere texto a partir do modelo
```
python language_generation
```

# Reference: https://medium.com/analytics-vidhya/build-a-simple-predictive-keyboard-using-python-and-keras-b78d3c88cffb