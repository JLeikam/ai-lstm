from nltk.tokenize import sent_tokenize, word_tokenize
import io
import random
import datetime

class MarkdownGenerator:
    text_file_name = ""
    markdown_file_name = ""
    model_name = ""
    book = ""
    author = ""
    sentence_index = 0

    def __init__(self, text_file_name, markdown_file_name, author, book, model_name):
        self.text_file_name = text_file_name
        self.markdown_file_name = markdown_file_name
        self.author = author
        self.book = book
        self.model_name = model_name

    def get_text(self):
        with io.open(self.text_file_name, mode='r', encoding='utf-8') as f:
            text = f.read().lower()

        return text

    def get_sentences(self):
        text = self.get_text()
        tokenized_sentences = sent_tokenize(text)
        for index, sentence in enumerate(tokenized_sentences):
            #remove first word of first sentence, or if it is not a word, delete it
            if(index == 0):
                if(' ' in sentence):
                    tokenized_sentences[index] = sentence.split(' ', 1)[1]
                else:
                    del tokenized_sentences[index]
                sentence = tokenized_sentences[index]
            # remove last word of last sentence
            if(index == len(tokenized_sentences)-1):
               tokenized_sentences[index] = sentence.rsplit(' ', 1)[0]
               print(tokenized_sentences[index])

        return tokenized_sentences

    def get_words(self):
        text = self.get_text()
        tokenized_words = word_tokenize(text)
        # remove first and last 'word' because it inevitably gets cut off in text generation
        del tokenized_words[0]
        del tokenized_words[-1]
        return tokenized_words

    def get_title(self):
        words = self.get_words()
        title = []
        for x in range(random.randint(3,9)):
            title.append(words[x])
        return title

    def get_author(self):
        return self.author

    def get_book(self):
        return self.book

    def get_model(self):
        return self.model_name

markDownGenerator = MarkdownGenerator("text_gen_output.txt", "test.md", "Kant", "A Critique of Pure Reason", "v1.06.1000.hdf5")
sentences = markDownGenerator.get_sentences()
title = markDownGenerator.get_title()
words = markDownGenerator.get_words()
# print(words)
print(sentences)
f = open("test.md", "w+", encoding="utf-8")
f.write('---\n')
f.write('title: ' + ' '.join(title) +'\n')
f.write('date: ' + str(datetime.date.today().strftime("%Y-%m-%d")) + '\n')
f.write('author: ' + markDownGenerator.get_author() + '\n')
f.write('book: ' + markDownGenerator.get_book() + '\n')
f.write('model: ' + markDownGenerator.get_model() + '\n')
f.write('---\n')
for sentence in sentences:
    new_sentence = sentence.capitalize().rjust(len(sentence)+1) # capitalize first letter of sentence and add space
    f.write(new_sentence)
f.write(".")
f.close()


#TODO:
#create title
#create timestamp/date
#create heading
#create paragraph

# add model metadata info
# average words per blog post: 1142
# average characters per word: 6
# average words per paragraph: 150
# average number of paragraphs per blog post: 7
# average number of headings per blog post: 3