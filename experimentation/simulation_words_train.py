# Created by Josh Leikam
# 3/20/19
# Capital Factory Project
# Prepares/trains RNN model
# based on tutorial from https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
import string
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


def get_text(filename):
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


def save_text(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# load file
my_filename = 'wonderland.txt'
text = get_text(my_filename)

# clean file
tokens = get_clean_tokens_from(text)


# organize text into sequences of tokens
length = 100 + 1
sequences = list()
for i in range(length, len(tokens)):
    # select a sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)

print('Total Sequences: %d' % len(sequences))

# save cleaned sequences to new file
filename_to_save = 'wonderland_sequences.txt'
save_text(sequences, filename_to_save)


# load
my_filename = 'wonderland_sequences.txt'
text = get_text(my_filename)
lines = text.split('\n')

# integer encode sequences of words

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1


sequences = array(sequences)

X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# define model
# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=150)

# save model
model.save('model_v3.h5')

# save tokenizer
dump(tokenizer, open('tokenizer_v3.pkl', 'wb'))

