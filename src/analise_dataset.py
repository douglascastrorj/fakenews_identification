#!/usr/bin/env python
# -*- coding: utf-8 -*-

import read_dataset as rd
from text_processor import Preprocessor

import spacy

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from lexicalrichness import LexicalRichness

print('loading spacy...')
nlp = spacy.load('en_core_web_sm')
processor = Preprocessor()

def getDataset(remove_stop = False):
    print('reading dataset...')
    folders_fake = ['../datasets/FakeNewsNet-master/Data/BuzzFeed/FakeNewsContent/', '../datasets/FakeNewsNet-master/Data/PolitiFact/FakeNewsContent/']
    folders_real = ['../datasets/FakeNewsNet-master/Data/BuzzFeed/RealNewsContent/', '../datasets/FakeNewsNet-master/Data/PolitiFact/RealNewsContent/']

    politic_fake = folders_fake[1]
    politic_real = folders_real[1]

    dataset_fake = [ processor.proccess_text(data['text'], remove_stop_words=remove_stop) for data in rd.read_folder(politic_fake)]
    dataset_real = [ processor.proccess_text(data['text'], remove_stop_words=remove_stop) for data in rd.read_folder(politic_real, 1, 'real')]

    # token.is_stop
    # if remove_stop:
    #     dataset_fake = [ token for token in dataset_fake if token.is_stop == False]
    #     dataset_real = [ token for token in dataset_real if token.is_stop == False]
    #     print (len(dataset_real))

    return dataset_fake, dataset_real

def getTopNWords(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def to_csv(data, path='output.csv'):
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path)

def count_adj(dataset):
    data = [ nlp(content) for content in dataset ]

    adj = []
    for item in data:
        adj = adj + [token.text.lower() for token in item if token.tag_ == 'JJ']
    #print(adj)

    freq = {}
    for jj in adj:
        freq[jj] = freq.get(jj, 0) + 1
    words_freq = sorted(freq.items(), key = lambda x: x[1], reverse=True)
    return words_freq[:20]

#posso usar .ent_type_
def count_ents(dataset, entType = 'GPE'):
    data = [ nlp(content) for content in dataset ]
    ents = []
    for item in data:
        ents = ents + [token.text for token in item if token.ent_type_ == entType]
    freq = {}
    for e in ents:
        freq[e] = freq.get(e, 0) + 1
    words_freq = sorted(freq.items(), key = lambda x: x[1], reverse=True)
    return words_freq[:20]

def lexRichess(dataset):
    lex = [ lexicalrichness(text) for text in dataset ]

    ttrs = [ l.ttr for l in lex ]
    mean_ttr = sum(ttrs)/len(ttrs)

    mltds = [ l.mtld(threshold=0.72) for l in lex ]
    mean_mltd = sum(mltds)/len(mltds)

    return mean_ttr, mean_mltd
    
    
def write(file, dataset):
    for text in dataset:
        file.write(text + "\n")


dataset_fake, dataset_real = getDataset(remove_stop = True)

fileFake = open('fakenews.txt', 'w', encoding='utf8')
fileTrust = open('trustnews.txt', 'w', encoding='utf8')

write(fileFake, dataset_fake)
write(fileTrust, dataset_real)


# adj_fake = count_adj(dataset_fake)
# adj_real = count_adj(dataset_real)
# # all_adj = count_adj(dataset_fake + dataset_real)

# # print('\ntop 20 adj')
# # print(adj_fake)
# # print(adj_real)


# # print('\ntop 20 ents')
# ents_fake = count_ents(dataset_fake)
# ents_real = count_ents(dataset_real)

# print(ents_fake)
# print(ents_real)

# print(lexRichess(dataset_fake))
# print(lexRichess(dataset_real))