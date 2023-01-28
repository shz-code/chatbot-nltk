import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# To tokenize data
def tokenizer(sentence):
    return nltk.word_tokenize(sentence)
# Stem data to root form
def stemming(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())
# To get words that appeared in the sentence
def bag_of_words(tokenized_sentence, words):
    tokenized_sentence = [stemming(i) for i in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for i, word in enumerate(words):
        if word in tokenized_sentence:
            bag[i] = 1.0

    return bag
