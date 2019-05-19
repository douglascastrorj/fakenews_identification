#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag_sents, pos_tag


class Preprocessor():
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        pass

    def process_dataset(self,dataset, remove_stop_words=False, stem=False, remove_ponctuation=False, n_gram=1, tags=False):
        return [ 
            self.proccess_text(
                                text,
                                remove_ponctuation=remove_ponctuation,
                                remove_stop_words=remove_stop_words,
                                stem=stem,
                                n_gram=n_gram,
                                tags=tags
                            )
            for text in dataset
        ]
    
    def proccess_text(self, text, remove_stop_words=False, stem=False, remove_ponctuation=False, n_gram=1, tags=False):
        words = word_tokenize(text)
        if remove_stop_words :
            words = [w for w in words if not w in self.stop_words]
        if stem:
            words = [ self.stemmer.stem(word) for word in words ]

        processed_text = ' '.join(words)

        if tags:
            try:
                processed_text = self.apply_pos_tag(processed_text)
            except:
                print 'Error on apply pos tag '
        if n_gram > 1:
            processed_text = self.n_gram(processed_text, n_gram)

        return processed_text
    
    def tokenize(self, text):
        return word_tokenize(text)
    
    def apply_pos_tag(self, text):
        words = text.strip().encode("utf-8").split(" ")
        tags = []
        for word in words:
            try:
               tags.append( pos_tag([word])[0][1] )
            except:
                pass
        gramatical_class = ['@feature_' + tag for tag in tags]
        result = words + gramatical_class
        return ' '.join(result)

    def n_gram(self, text, n=2):
        words = text.split(' ')
        model = []
        for i in range(0, len(words) - 1):
            model.append(words[i])
            gram = '@featureNGram'
            for j in range(0, n):
                if(i + j < len(words)):
                    gram = gram + '_' + words[i+j]
            model.append(gram)
        model.append(words[len(words) - 1])
        return ' '.join(model)

# data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."

# data = 'hello my old friend, are you working so much?'
# preprocessor = Preprocessor()
# print preprocessor.n_gram(data, n=2)
# print preprocessor.proccess_text(data, n_gram=2, tags=True, stem=True)
# print preprocessor.apply_pos_tag(data)

# dataset = [
# 'Today was a bad day',
# 'I love running in the park',
# 'I used to have a cat when I was a kid'
# ]

# print preprocessor.process_dataset(dataset)