# Created by Josh Leikam
# 3/20/19
# Capital Factory Project
# loads saved RNN model

from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def get_text(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    my_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([my_text])[0]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        my_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


my_filename = 'wonderland_sequences.txt'
text = get_text(my_filename)
lines = text.split('\n')
seq_length = len(lines[0].split()) - 1

model = load_model('model_v3.h5')

tokenizer = load(open('tokenizer_v3.pkl', 'rb'))

seed_text = lines[randint(0, len(lines))]
print(seed_text + '\n')

generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
print(generated)


