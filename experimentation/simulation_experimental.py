# Josh Leikam
# Created 3/21/19
# Capital Factory Project
# Word level RNN


# keras module for building LSTM

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.models import Sequential
import keras.utils as ku


# set seeds for reproducability
from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

SEQUENCE_LENGTH = 20
DATASET_FILENAME = "wonderland.txt"
CORPUS_FILENAME = "wonderland_sequences.txt"

# load the dataset

def get_text_from_file(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def get_clean_tokens_from(text):
    text = text.replace('--', ' ')
    tokens = text.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # convert to lowercase
    tokens = [word.lower() for word in tokens]
    return tokens

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt

def save_text(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def get_sequences_of_tokens(length, tokens):
    length += 1
    sequences = list()
    for i in range(length, len(tokens)):
        # select a sequence of tokens
        seq = tokens[i - length:i]
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)
    return sequences

def get_sequence_of_tokens(lines):
    tokenizer.fit_on_texts(lines)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in lines:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range( 1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    return input_sequences, total_words


def generate_padded_sequences(input_sequences):
    max_sequence_length = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_length


def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1

    model = Sequential()

    model.add(Embedding(total_words, 10, input_length=input_len))

    model.add(LSTM(100))
    model.add(Dropout(0.1))

    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# load data set text
data_text = get_text_from_file(DATASET_FILENAME)

# tokenize text
tokens = get_clean_tokens_from(data_text)

# organize text into sequences of tokens
sequences = get_sequences_of_tokens(SEQUENCE_LENGTH, tokens)

# save cleaned sequences to new file
save_text(sequences, CORPUS_FILENAME)

# load sequences file
corpus = get_text_from_file(CORPUS_FILENAME)
lines = corpus.split('\n')

print(lines[:10])

### EXPIREMENTAL ###

tokenizer = Tokenizer()
inp_sequences, total_words = get_sequence_of_tokens(lines)

predictors, label, max_sequence_length = generate_padded_sequences(inp_sequences)

# model = create_model(max_sequence_length, total_words)
# model.summary()
#
# # train
# model.fit(predictors, label, epochs=3)
#
# model.save('model_experimental.h5')

model = load_model('model_experimental.h5')

predicted_text = generate_text("hello", 50, model, max_sequence_length)
print(predicted_text)