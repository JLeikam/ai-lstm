# Josh Leikam
# 3/22/19
# Simulating the Minds of Philosophers Project
# Based on:
# https://gilberttanner.com/2018/08/03/keras-tutorial-4-lstm-text-generation/
# https://keras.io/examples/lstm_text_generation/
# https://medium.freecodecamp.org/applied-introduction-to-lstms-for-text-generation-380158b29fb3
""" Reccomended note from Keras documentation:
If you try this script on new data, make sure your corpus has at least ~100k characters. ~1M is better."""

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import io
import sys


FILE_NAME = "kant.txt"
# MODEL_NAME = "model_human_256_200.h5"
MODEL_NAME = "model_kant.h5"
# CHECKPOINT_FILEPATH = "weights/human/weights.test.{epoch:02d}.hdf5"
CHECKPOINT_FILEPATH = "models/kant/v1/model.v1.06.{epoch:02d}.hdf5"

# load the text into memory
with io.open(FILE_NAME, mode='r', encoding='utf-8') as f:
    text = f.read().lower()

# text = open(FILE_NAME, 'r').read().lower()

# get all unique characters
chars = sorted(list(set(text)))


# our model cannot be trained on characters directly, so we create two dictionaries
# the first dictionary maps our characters to integers so that our model can 'read' it
# the second dictionary maps our integers back into characters so that humans can read the generated text

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# now we need to create features and labels
# more on features: https://en.wikipedia.org/wiki/Feature_(machine_learning)
# more on labels: https://en.wikipedia.org/wiki/Labeled_data
# we will use subsequences of text as the features for our model
# we will use the next character as our label

max_len = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - max_len, step):
    sentences.append(text[i: i + max_len])
    next_chars.append(text[i + max_len])

# vectorize our data
x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build simple model
# model = Sequential()
# model.add(LSTM(128, input_shape=(max_len, len(chars))))
# model.add(Dense(len(chars)))
# model.add(Activation('softmax'))

model = load_model("models/kant/v1/model.v1.05.1000.hdf5")

# load weights
# model.load_weights(CHECKPOINT_FILEPATH)

# build sophisticated model
# model = Sequential()
# model.add(LSTM(128, input_shape=(max_len, len(chars))))
# model.add(Dense(len(chars), activation='softmax'))

# compile model
# optimizer = RMSprop(lr=0.00001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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

# helper function
# generates text after each epoch
# selects random starting index then creates a text of 400 chars for 5 different temperatures
def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text. Logs text
    f = open("kant_v1.06.txt", "a", encoding="utf-8")
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    f.write('\n')
    f.write('----- Generating text after Epoch: %d' % epoch)
    f.write('\n')

    start_index = random.randint(0, len(text) - max_len - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)
        f.write('----- diversity:' + str(diversity))

        generated = ''
        sentence = text[start_index: start_index + max_len]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
        f.write('----- Generating with seed: "' + sentence + '"')
        f.write(generated)
        f.write('\n')

        for i in range(400):
            x_pred = np.zeros((1, max_len, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            f.write(next_char)
            sys.stdout.flush()
        print()
    f.close()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
checkpoint = ModelCheckpoint(CHECKPOINT_FILEPATH, monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, min_lr=0.001)

callbacks = [print_callback, checkpoint]

model.fit(x, y, batch_size=128, epochs=1000, callbacks=callbacks)

# save most recent model
model.save(MODEL_NAME)
