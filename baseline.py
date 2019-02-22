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

preprocessor = Preprocessor()


## LENDO DATASET        ######################
train,test = rd.read(percent_train=.5)
categories = ['fake', 'real']

train_text = rd.get_text(train)
# train_text = preprocessor.process_dataset(train_text)
train_target = rd.get_target(train)

test_text = rd.get_text(test)
# test_text = preprocessor.process_dataset(test_text)
test_target = rd.get_target(test)
#################################################


##              TREINANDO NAIVE               ##

print ('Treinando modelo com Naive bayes...')
text_clf = Pipeline([('vect', CountVectorizer()),
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

fake = 'Se ele criou o Kit gay para crianças de 6 anos de idade como ministro da  Educação de Lula imagine a imoralidade que esse homem faria como presidente do nosso país. Contra um governo imoral e em favor do futuro de nossas crianças Haddad você tem meu Desprezo!!!! #EleJamais'

texto_informal = 'caramba não sei que tipo de pessoa faria isso'
texto_formal = 'O senado aprovou.'
print(text_clf.predict([texto_informal, fake]) )
# print(text_clf.predict([texto_formal]) )

##              TREINANDO SVM               ##

# print ('\n\nTreinando modelo com SVM...')
# text_clf = Pipeline([('vect', CountVectorizer()),
#                       ('tfidf', TfidfTransformer()),
#                       ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                                             alpha=1e-3, random_state=42,
#                                             max_iter=5, tol=None)),
# ])
# text_clf.fit(train_text, train_target)
# print( 'Treino concluido.')

# predicted = text_clf.predict(test_text)
# print(np.mean(predicted == test_target) )

# #metricas
# print(metrics.classification_report(test_target, predicted,target_names=categories))
# print(metrics.confusion_matrix(test_target, predicted))
