#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag_sents, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from lexicalrichness import LexicalRichness
from itertools import groupby
import spacy
import numpy as np


# https://pypi.org/project/lexicalrichness/

def lexical_diversity(text):
    return len(set(text)) / len(text)
 
def words(entry):
    return filter(lambda w: len(w) > 0, [w.strip("0123456789!:,.?(){}[]") for w in entry.split()])
# https://swizec.com/blog/measuring-vocabulary-richness-with-python/swizec/2528
def yule(entry):
    # yule's I measure (the inverse of yule's K measure)
    # higher number is higher diversity - richer vocabulary
    d = {}
    stemmer = PorterStemmer()
    for w in words(entry):
        w = stemmer.stem(w).lower()
        try:
            d[w] += 1
        except KeyError:
            d[w] = 1
 
    M1 = float(len(d))
    M2 = sum([len(list(g))*(freq**2) for freq,g in groupby(sorted(d.values()))])
 
    try:
        return (M1*M1)/(M2-M1)
    except ZeroDivisionError:
        return 0

class Preprocessor():
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.nlp = spacy.load('en_core_web_lg')

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
                            ent=False,
                            sentiment=False,
                            vectorizer='count'
        ):

        # return processed_corpus
        if vectorizer == 'tfidf':
            # self.vectorizer = TfidfVectorizer( ngram_range=(1, n_gram), max_df=0.5, min_df=2 )
            self.vectorizer = TfidfVectorizer()
            # n_gram = 1
        else:
            self.vectorizer = CountVectorizer()
        
        processed_corpus = [ 
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
                                ent=ent,
                                sentiment=sentiment
                            )
            for text in dataset
        ]

        lex_features = [ ]
        for text in dataset:
            lex = LexicalRichness(text)
            li = []
            try:
                li.append(lex.ttr)
            except:
                li.append(0.0)
            try:
                li.append(lex.rttr)
            except:
                li.append(0.0)
            try:
                li.append(lex.cttr)
            except:
                li.append(0.0)
            try:
                li.append(lex.mtld(threshold=0.72))
            except:
                li.append(0.0) 
            lex_features.append(li)
        lex_features = np.array(lex_features)

        X = self.vectorizer.fit_transform(processed_corpus)
        X = X.toarray()
        X = np.concatenate((X, lex_features), axis=1)

        # print(len(self.vectorizer.get_feature_names()), '- Vocabulary\n\n')
        return X
    
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
                        ent=False,
                        sentiment=False
        ):

        features = []
        n_grams = []
        tokens = self.nlp(text)
        if remove_punct:
            tokens = [ token for token in tokens if token.dep_ != 'punct']
        if remove_stop_words:
            tokens = [token for token in tokens if token.is_stop == False]
        if n_gram > 1:
            n_grams = n_grams + self.n_gram(tokens, n_gram)
            
        if ent:
            features = features + self.extract_ents(self.nlp(text))
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
    
    def extract_sentiment(self, doc):
        sentiments = []
        for token in doc:
            sentiments.append(token.sentiment)
        return sentiments
    
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

# data = 'hello my old friend, are you working so much man??'

# corpus = [
#     'I woke up this morning to find a variation of this headline splashed all over my news feed:\n\nBill Clinton: \u2018Natural\u2019 For Foundation Donors to Seek Favors\n\nHere\u2019s Google:\n\nNaturally, my reaction was \u201coh, s**t, what did Bill Clinton do to damage his wife\u2019s campaign now?\u201d\n\nOf course, the headline sounds really, really awful. It plays right into the idea that the Clinton Foundation is all about pay to play, just like Donald Trump has been saying all along. Unfortunately, it takes reading beyond the headlines, which is something most people don\u2019t do, to find out the real story \u2013 and the real story is that there is no pay to play.\n\n\u201cIt was natural for people who\u2019ve been our political allies and personal friends to call and ask for things. And I trusted the State Department wouldn\u2019t do anything they shouldn\u2019t do,\u201d Clinton told NPR in an interview that aired Monday morning. Source: CNN\n\nIn other words, people can ask for favors, but that certainly doesn\u2019t mean they\u2019ll get them. Leaked emails have shown that some Clinton Foundation donors have gotten meetings with Clinton and that others were turned down. There is zero evidence of pay to play. In other words, people might have asked for favors, but there\u2019s no evidence they got them.\n\nNow, let\u2019s talk about the foundation the media doesn\u2019t like to mention, the Trump Foundation. Trump hasn\u2019t given to his own foundation since 2008. He does collect money from others, though, and gives it in his name. He also takes from the charity and allegedly buys things like oil paintings and football helmets, all for himself, but out of charity money.\n\nNew York Attorney General Eric Schneiderman said in a September 13 CNN interview that his office is investigating Trump\u2019s charitable foundation over concerns that it \u201cengaged in some impropriety\u201d as related to New York charity laws. The investigation launched amid reports from The Washington Post that Trump spent money from his charity on items meant to benefit himself, such as a $20,000 oil painting of himself and a $12,000 autographed football helmet, and also recycled others\u2019 contributions \u201cto make them appear to have come from him\u201d although he \u201chasn\u2019t given to the foundation since 2008.\u201d Source: Media Matters\n\nMedia Matters goes on to talk about the double standard and',
#     'hello my old friend, are you working so much man??',
#      "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
# ]
# preprocessor = Preprocessor()
# doc = preprocessor.nlp(corpus[0])
# print(preprocessor.extract_sentiment( doc ))
# print( 
#     preprocessor.process_dataset(
#         corpus, 
#         n_gram = 2,
#         pos=True,
#         tags=True,
#         dep=True,
#         stem = True,
#         remove_stop_words = True,
#         remove_punct = True,
#         ent=True,
#         alpha=True
#     ) 
# )
# print( preprocessor.n_gram(preprocessor.nlp(data), n=2))
# print( preprocessor.proccess_text(data, n_gram=2, tags=True, stem=True, remove_punct=True))
# print preprocessor.apply_pos_tag(data)

# dataset = [
# 'Today was a bad day',
# 'I love running in the park',
# 'I used to have a cat when I was a kid'
# ]

# print preprocessor.process_dataset(dataset)