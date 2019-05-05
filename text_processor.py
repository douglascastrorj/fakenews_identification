#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag_sents


class Preprocessor():
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        pass

    def process_dataset(self,dataset):
        return [ self.proccess_text(text.decode('utf-8')) for text in dataset]
    
    def proccess_text(self, text):

        words = word_tokenize(text)
        filtered_sentence = [w for w in words if not w in self.stop_words]
        stemmed_words = [ self.stemmer.stem(word) for word in filtered_sentence ]
        return ' '.join(stemmed_words)
    
    def tokenize(self, text):
        return word_tokenize(text)

    def pos_tag(self, tokens):
        return pos_tag_sents(tokens)
    
    def n_gram(self, text, n=2):
        words = text.split(' ')
        model = []
        for i in range(0, len(words) - 1):
            model.append(words[i])
            gram = ''
            for j in range(0, n):
                if(i + j < len(words)):
                    gram = gram + '_@_' + words[i+j]
            model.append(gram)
        model.append(words[len(words) - 1])
        return ' '.join(model)

# data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."

# data = 'oi amigo legal xD'
# preprocessor = Preprocessor()
# print preprocessor.n_gram(data, n=2)

# dataset = [
# 'Today was a bad day',
# 'I love running in the park',
# 'I used to have a cat when I was a kid'
# ]

# print preprocessor.process_dataset(dataset)