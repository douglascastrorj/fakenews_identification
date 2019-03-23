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

## LENDO DATASET        ######################
train,test = rd.read(percent_train=.5)
categories = ['fake', 'real']

train_text = rd.get_text(train)

train_target = rd.get_target(train)

test_text = rd.get_text(test)

test_target = rd.get_target(test)
#################################################


##              TREINANDO NAIVE               ##

print ('Treinando modelo com Naive bayes...')
text_clf = Pipeline([
                    ('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
])
text_clf.fit(train_text , train_target)
print( 'Treino concluido.')

#avaliacao de desempenho no conjunto de teste
predicted = text_clf.predict(test_text)
print('acuracia ', np.mean(predicted == test_target) )


#metricas
print(metrics.classification_report(test_target, predicted,target_names=categories))
print(metrics.confusion_matrix(test_target, predicted))
