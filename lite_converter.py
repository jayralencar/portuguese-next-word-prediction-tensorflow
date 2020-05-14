# from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import load_model


model = load_model('saved_models/word_prediction.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open("saved_models/converted_model.tflite", "wb").write(tflite_model)
