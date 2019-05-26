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
from sklearn.model_selection import cross_val_score, cross_validate, ShuffleSplit

from sklearn.metrics import recall_score

import read_dataset as rd
from text_processor import Preprocessor

### PROCESSANDO TEXTO ######

def analisar_features(
  train_text,
  n_gram = 0,
  pos=False,
  tags=False,
  dep=False,
  stem = False,
  remove_stop_words = False,
  remove_punct = False,
  ent=False,
  alpha=False
):

  print('Features utilizadas: \n' )
  print('NGRAM: '+ str(n_gram) + '\n' )
  print('tags: '+ str(tags) + '\n' )
  print('pos: '+ str(pos) + '\n' )
  print('dep: '+ str(dep) + '\n' )
  print('stem: '+ str(stem) + '\n' )
  print('ent: '+ str(ent) + '\n' )
  print('alpha: '+ str(alpha) + '\n' )
  print('Remove stopwords: '+ str(remove_stop_words) + '\n' )
  print('Remove ponctuation: '+ str(remove_punct) + '\n\n' )

  print('Processando texto...')

  processor = Preprocessor()
  train_text =  processor.process_dataset(
                  train_text, 
                  n_gram=n_gram, 
                  stem=stem, 
                  tags=tags,
                  remove_stop_words=remove_stop_words, 
                  remove_punct=remove_punct,
                  pos=pos,
                  dep=dep,
                  alpha=alpha
                )
  # test_text = processor.process_dataset(
  #               test_text,
  #               n_gram=n_gram, 
  #               stem=stem,
  #               tags=tags,
  #               remove_stop_words=remove_stop_words, 
  #               remove_punct=remove_punct,
  #               pos=pos,
  #               dep=dep
  #             )

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

  scoring = ['precision_macro', 'recall_macro', 'f1_macro']
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0 )
  scores = cross_validate(text_clf, train_text, train_target, cv=5, scoring=scoring)
  print(scores['test_precision_macro'])
  print(scores['test_recall_macro'])
  print(scores['test_f1_macro'])

  print("Accuracy: %0.2f (+/- %0.2f)" % (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std() * 2))

  print('Escrevendo arquivo de log\n')
  file = open('log_features.txt', 'a')

  file.write('Features utilizadas: \n' )
  file.write('NGRAM: '+ str(n_gram) + '\n' )
  file.write('pos: '+ str(pos) + '\n' )
  file.write('dep: '+ str(dep) + '\n' )
  file.write('tags: '+ str(tags) + '\n' )
  file.write('stem: '+ str(stem) + '\n' )
  file.write('ent: '+ str(ent) + '\n' )
  file.write('alpha: '+ str(alpha) + '\n' )
  file.write('Remove stopwords: '+ str(remove_stop_words) + '\n' )
  file.write('Remove ponctuation: '+ str(remove_punct) + '\n\n' )

  file.write('Recall Macro: ' + str(scores['test_recall_macro'].mean()) + ' (+/-) ' + str(scores['test_recall_macro'].std() * 2) + '\n' )
  file.write('Precision Macro: ' + str(scores['test_precision_macro'].mean()) + ' (+/-) ' + str(scores['test_precision_macro'].std() * 2) + '\n' )
  file.write('F1 Macro: ' + str(scores['test_f1_macro'].mean()) + ' (+/-) ' +str(scores['test_f1_macro'].std() * 2) + '\n' )

  file.write('\n\n#############################################\n\n')
  file.close() 



## LENDO DATASET        ######################
train,test = rd.read()
categories = ['fake', 'real']

train_text = rd.get_text(train)
train_target = rd.get_target(train)

# test_text = rd.get_text(test)
# test_target = rd.get_target(test)
#################################################

remove_stop_words=False
stem=False
remove_punct=False
n_gram=1
tags=False
pos=False
dep=False
alpha=False
ent=False


combinations = [
  {
    'remove_stop_words':False,
    'stem':False,
    'remove_punct':False,
    'n_gram':1,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':True,
    'stem':False,
    'remove_punct':False,
    'n_gram':1,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':False,
    'stem':True,
    'remove_punct':False,
    'n_gram':1,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':False,
    'stem':False,
    'remove_punct':True,
    'n_gram':1,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':False,
    'stem':False,
    'remove_punct':False,
    'n_gram':2,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':False,
    'stem':False,
    'remove_punct':False,
    'n_gram':1,
    'tags':True,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':False,
    'stem':False,
    'remove_punct':False,
    'n_gram':1,
    'tags':False,
    'pos':True,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':False,
    'stem':False,
    'remove_punct':False,
    'n_gram':1,
    'tags':False,
    'pos':False,
    'dep':True,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':False,
    'stem':False,
    'remove_punct':False,
    'n_gram':1,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':True,
    'ent':False
  },
  {
    'remove_stop_words':False,
    'stem':False,
    'remove_punct':False,
    'n_gram':1,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':True
  },
  # dois ao mesmo tempo
  {
    'remove_stop_words':True,
    'stem':True,
    'remove_punct':False,
    'n_gram':1,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':True,
    'stem':False,
    'remove_punct':True,
    'n_gram':1,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':True,
    'stem':False,
    'remove_punct':False,
    'n_gram':2,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':True,
    'stem':False,
    'remove_punct':False,
    'n_gram':1,
    'tags':True,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':True,
    'stem':False,
    'remove_punct':False,
    'n_gram':1,
    'tags':False,
    'pos':True,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':True,
    'stem':False,
    'remove_punct':False,
    'n_gram':1,
    'tags':False,
    'pos':False,
    'dep':True,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':True,
    'stem':False,
    'remove_punct':False,
    'n_gram':1,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':True,
    'ent':False
  },
  {
    'remove_stop_words':True,
    'stem':False,
    'remove_punct':False,
    'n_gram':1,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':True
  },
  {
    'remove_stop_words':False,
    'stem':True,
    'remove_punct':True,
    'n_gram':1,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':False,
    'stem':True,
    'remove_punct':False,
    'n_gram':2,
    'tags':False,
    'pos':False,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':False,
    'stem':True,
    'remove_punct':False,
    'n_gram':1,
    'tags':True,
    'pos':True,
    'dep':False,
    'alpha':False,
    'ent':False
  },
  {
    'remove_stop_words':False,
    'stem':True,
    'remove_punct':False,
    'n_gram':1,
    'tags':True,
    'pos':True,
    'dep':False,
    'alpha':False,
    'ent':False
  },
]


for combination in combinations:
  analisar_features(train_text,
                    stem=combination['stem'],
                    remove_stop_words=combination['remove_stop_words'], 
                    remove_punct=combination['remove_punct'], 
                    n_gram=combination['n_gram'], 
                    tags=combination['tags'], 
                    pos=combination['pos'], 
                    dep=combination['dep'], 
                    alpha=combination['alpha'],
                    ent=combination['ent']
                    )



#avaliacao de desempenho no conjunto de teste
# predicted = text_clf.predict(test_text)
# print('acuracia ', np.mean(predicted == test_target) )

# Compute the precision
# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. 
# The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

# Compute the recall
# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
# The recall is intuitively the ability of the classifier to find all the positive samples.
# The best value is 1 and the worst value is 0.
#metricas
# print(metrics.classification_report(test_target, predicted,target_names=categories))
# print(metrics.f1_score(test_target, predicted))
# print(metrics.precision_score(test_target, predicted))
# print(metrics.recall_score(test_target, predicted))
# print(metrics.confusion_matrix(test_target, predicted))

