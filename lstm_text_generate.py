# Josh Leikam
# 3/22/19
# Simulating the Minds of Philosophers Project
# Based on:
# https://gilberttanner.com/2018/08/03/keras-tutorial-4-lstm-text-generation/
# https://keras.io/examples/lstm_text_generation/
# TODO: make an automated turing test generator, game to play which is real Kant vs Kant's neural network

from keras.models import load_model
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import random
import io
import numpy as np

FILE_NAME = "kant.txt"
MODEL_NAME = "models/kant.model.v1.06.1000.hdf5"
# MODEL_WEIGHTS_NAME = "weights/weights.36.hdf5"

# used to load a previously trained model
model = load_model(MODEL_NAME)


# load the text into memory
# text = open(FILE_NAME, 'r').read().lower().encode('utf-8')
# load the text into memory
with io.open(FILE_NAME, mode='r', encoding='utf-8') as f:
    text = f.read().lower()

# get all unique characters
chars = sorted(list(set(text)))
# def max length
max_len = 40




# model = Sequential()
# model.add(LSTM(128, input_shape=(max_len, len(chars))))
# model.add(Dense(len(chars), activation='softmax'))
# used when wanting to load weights from a previous model checkpoint
# model.load_weights(MODEL_WEIGHTS_NAME)

# our model cannot be trained on characters directly, so we create two dictionaries
# the first dictionary maps our characters to integers so that our model can 'read' it
# the second dictionary maps our integers back into characters so that humans can read the generated text

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# helper function
# gets index of predicted char
# raising temperature value increases randomness/wildness of predicted characters
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(length, diversity):
    # Get random starting text

    start_index = random.randint(0, len(text) - max_len - 1)
    print(start_index)
    generated = ''
    sentence = text[start_index: start_index + max_len]
    print(sentence)
    generated += sentence
    for i in range(length):
            x_pred = np.zeros((1, max_len, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
    return generated


print(generate_text(300, 0.5))