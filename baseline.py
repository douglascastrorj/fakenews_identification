#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import numpy as np
from sklearn.datasets import fetch_20newsgroups

import read_dataset as rd
from text_processor import Preprocessor

## LENDO DATASET        ######################
train,test = rd.read(percent_train=.7)
categories = ['fake', 'real']

train_text = rd.get_text(train)

train_target = rd.get_target(train)

test_text = rd.get_text(test)

test_target = rd.get_target(test)
#################################################

### PROCESSANDO TEXTO ######

print('Processando texto...')
n_gram = 0
pos=False
tags=False
dep=False
stem = False
remove_stop_words = False
remove_punct = False

processor = Preprocessor()
train_text =  processor.process_dataset(
                train_text, 
                n_gram=n_gram, 
                stem=stem, 
                tags=tags,
                remove_stop_words=remove_stop_words, 
                remove_punct=remove_punct,
                pos=pos,
                dep=dep
              )

test_text = processor.process_dataset(
              test_text,
              n_gram=n_gram, 
              stem=stem,
              tags=tags,
              remove_stop_words=remove_stop_words, 
              remove_punct=remove_punct,
              pos=pos,
              dep=dep
            )

##              TREINANDO NAIVE               ##

print ('Treinando modelo...')
text_clf = Pipeline([
                    ('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
                      # ('clf', SGDClassifier(loss='hinge', penalty='l2',
                      #                       alpha=1e-3, random_state=42,
                      #                       max_iter=7, tol=None)),
])
text_clf.fit(train_text , train_target)
print( 'Treino concluido.')

#avaliacao de desempenho no conjunto de teste
predicted = text_clf.predict(test_text)
print('acuracia ', np.mean(predicted == test_target) )

# Compute the precision
# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. 
# The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

# Compute the recall
# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
# The recall is intuitively the ability of the classifier to find all the positive samples.
# The best value is 1 and the worst value is 0.
#metricas
print(metrics.classification_report(test_target, predicted,target_names=categories))
print(metrics.f1_score(test_target, predicted))
print(metrics.precision_score(test_target, predicted))
print(metrics.recall_score(test_target, predicted))
print(metrics.confusion_matrix(test_target, predicted))

# print('Escrevendo arquivo de log\n')
# file = open('log.txt', 'a')

# file.write('Features utilizadas: \n' )
# file.write('NGRAM: '+ str(n_gram) + '\n' )
# file.write('tags: '+ str(tags) + '\n' )
# file.write('stem: '+ str(stem) + '\n' )
# file.write('Remove stopwords: '+ str(remove_stop_words) + '\n' )
# file.write('Remove ponctuation: '+ str(remove_punct) + '\n' )
# file.write('\nAcuracia: ' + str(np.mean(predicted == test_target)) )
# file.write('\n')
# file.write(metrics.classification_report(test_target, predicted,target_names=categories))
# file.write('\n')
# file.write('Matriz de Confusao: \n\n')
# file.write(np.array2string(metrics.confusion_matrix(test_target, predicted), separator=', '))
# file.write('\n\n#############################################\n\n')
# file.close() 