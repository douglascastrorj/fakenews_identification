#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag_sents, pos_tag
import spacy

class Preprocessor():
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.nlp = spacy.load('en_core_web_sm')
        pass

    def process_dataset(    self,
                            dataset, 
                            remove_stop_words=False, 
                            stem=False, 
                            remove_punct=False, 
                            n_gram=1, 
                            tags=False, 
                            pos=False, 
                            dep=False, 
                            alpha=False,
                            ent=False
        ):
        return [ 
            self.proccess_text(
                                text,
                                remove_stop_words=False, 
                                stem=stem, 
                                remove_punct=remove_punct, 
                                n_gram=n_gram, 
                                tags=tags, 
                                pos=pos, 
                                dep=dep, 
                                alpha=alpha,
                                ent=ent
                            )
            for text in dataset
        ]
    
    def proccess_text(  self, 
                        text, 
                        remove_stop_words=False, 
                        stem=False, 
                        remove_punct=False, 
                        n_gram=1, 
                        tags=False, 
                        pos=False, 
                        dep=False, 
                        alpha=False,
                        ent=False
        ):

        features = []
        n_grams = []
        tokens = self.nlp(text)
        if remove_punct:
            tokens = [ token for token in tokens if token.dep_ != 'punct']
        if remove_stop_words:
            tokens = [token for token in tokens if token.is_stop == False]
        if n_gram > 1:
            for i in range(2, n_gram):
                n_grams = n_grams + self.n_gram(tokens, n_gram)
            
        if ent:
            features.append(self.extract_ents(tokens))
        for token in tokens:
            features.append(token.text)

        for token in tokens:
            if stem:
                features.append('@feature_lemma_'+token.lemma_)
            if pos:
                features.append('@feature_pos_'+token.pos_)
            if tags:
                features.append( '@feature_tag_'+token.tag_)
            if dep:
                features.append('@feature_dep_'+ token.dep_)
            if alpha:
                # features.append('@feature_shape_'+token.shape_)
                features.append('@feature_alpha_'+str(token.is_alpha))
                # features.append('@feature_stop_'+ str(token.is_stop))
        features = features + n_grams
        return ' '.join(features)
    
    def extract_ents(self, doc):
        ents = []
        for ent in doc.ents:
            ents.append(ent.label_)
        return ents

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

    def n_gram(self, tokens, n=2):
        n_grams = []
        for i in range(0, len(tokens) - 1):
            n_grams.append(tokens[i].text)
            gram = '@feature_NGram'
            for j in range(0, n):
                if(i + j < len(tokens)):
                    gram = gram + '_' + tokens[i+j].text
            n_grams.append(gram)
        if len(tokens) > 1:
            n_grams.append(tokens[len(tokens) - 1].text)
        return n_grams

# data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."

# data = 'hello my old friend, are you working so much?? 234 1% ¨7 ** ( '
# preprocessor = Preprocessor()
# print( preprocessor.n_gram(preprocessor.nlp(data), n=2))
# print( preprocessor.proccess_text(data, n_gram=2, tags=True, stem=True, remove_punct=True))
# print preprocessor.apply_pos_tag(data)

# dataset = [
# 'Today was a bad day',
# 'I love running in the park',
# 'I used to have a cat when I was a kid'
# ]

# print preprocessor.process_dataset(dataset)