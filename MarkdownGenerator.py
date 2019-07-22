from nltk.tokenize import sent_tokenize, word_tokenize
import io
import random

class MarkdownGenerator:
    text_file_name = ""
    markdown_file_name = ""
    sentence_index = 0

    def __init__(self, text_file_name, markdown_file_name):
        self.text_file_name = text_file_name
        self.markdown_file_name = markdown_file_name

    def get_text(self):
        with io.open(self.text_file_name, mode='r', encoding='utf-8') as f:
            text = f.read().lower()

        return text

    def get_sentences(self):
        text = self.get_text()
        tokenized_sentences = sent_tokenize(text)
        return tokenized_sentences

    def get_words(self):
        text = self.get_text()
        tokenized_words = word_tokenize(text)
        return tokenized_words

    def get_title(self):
        words = self.get_words()
        title = []
        for x in range(random.randint(3,9)):
            title.append(words[x])
        return title



markDownGenerator = MarkdownGenerator("text_gen_output.txt", "test.md")
sentences = markDownGenerator.get_sentences()
print(sentences)
f = open("test.md", "a", encoding="utf-8")
for sentence in sentences:
    new_sentence = sentence.capitalize().rjust(len(sentence)+1) # capitalize first letter of sentence and add space
    f.write(new_sentence)
f.close()


#TODO:
#create title
#create timestamp/date
#create heading
#create paragraph

# average words per blog post: 1142
# average characters per word: 6
# average words per paragraph: 150
# average number of paragraphs per blog post: 7
# average number of headings per blog post: 3