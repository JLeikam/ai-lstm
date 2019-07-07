# Generates text for a trained lstm
#
#

from keras.models import load_model
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import random
import io
import numpy as np


class TextGenerator:

    max_len = 40
    # our model cannot be trained on characters directly, so we create two dictionaries
    # the first dictionary maps our characters to integers so that our model can 'read' it
    # the second dictionary maps our integers back into characters so that humans can read the generated text
    char_indices = {}
    indices_char = {}
    chars = ""

    def __init__(self, file_name, model_name):
        self.file_name = file_name
        self.model_name = model_name

    def load_keras_model(self):
        return load_model(self.model_name)

    def  load_file_text(self):
        with io.open(self.file_name, mode='r', encoding='utf-8') as f:
            text = f.read().lower()

        return text

    def get_characters_from_text(self):
        text = self.load_file_text()
        chars = sorted(list(set(text)))
        return chars

    def init_char_maps(self):
        chars = self.get_characters_from_text()
        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_char = dict((i, c) for i, c in enumerate(chars))
        self.chars = chars

    # helper function
    # gets index of predicted char
    # raising temperature value increases randomness/wildness of predicted characters
    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_text(self, length, diversity):
        # Get random starting text
        text = self.load_file_text()
        model = self.load_keras_model()
        start_index = random.randint(0, len(text) - self.max_len - 1)
        print(start_index)
        generated = ''
        sentence = text[start_index: start_index + self.max_len]
        print(sentence)
        generated += sentence
        for i in range(length):
            x_pred = np.zeros((1, self.max_len, len(self.chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, diversity)
            next_char = self.indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
        return generated

FILE_NAME = "ebooks/kant.txt"
MODEL_NAME = "models/kant.model.v1.06.1000.hdf5"
gen = TextGenerator(FILE_NAME, MODEL_NAME)
gen.get_characters_from_text()
gen.init_char_maps()
print(gen.generate_text(300, 0.5))
