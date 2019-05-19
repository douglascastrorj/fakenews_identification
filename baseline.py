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

for i in range(0,10):

  print('Treino: #',i + 1,'\n')

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
  n_gram = 3
  tags=False
  stem = True
  remove_stop_words = True
  remove_ponctuation = False

  processor = Preprocessor()
  train_text =  processor.process_dataset(
                  train_text, 
                  n_gram=n_gram, 
                  stem=stem, 
                  tags=tags,
                  remove_stop_words=remove_stop_words, 
                  remove_ponctuation=remove_ponctuation
                )

  test_text = processor.process_dataset(
                test_text,
                n_gram=n_gram, 
                stem=stem,
                tags=tags,
                remove_stop_words=remove_stop_words, 
                remove_ponctuation=remove_ponctuation
              )

  ##              TREINANDO NAIVE               ##

  print ('Treinando modelo...')
  text_clf = Pipeline([
                      ('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        # ('clf', MultinomialNB()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                              alpha=1e-3, random_state=42,
                                              max_iter=7, tol=None)),
  ])
  text_clf.fit(train_text , train_target)
  print( 'Treino concluido.')

  #avaliacao de desempenho no conjunto de teste
  predicted = text_clf.predict(test_text)
  print('acuracia ', np.mean(predicted == test_target) )


  #metricas
  print(metrics.classification_report(test_target, predicted,target_names=categories))
  print(metrics.confusion_matrix(test_target, predicted))

  print('Escrevendo arquivo de log\n')
  file = open('log.txt', 'a')

  file.write('Features utilizadas: \n' )
  file.write('NGRAM: '+ str(n_gram) + '\n' )
  file.write('tags: '+ str(tags) + '\n' )
  file.write('stem: '+ str(stem) + '\n' )
  file.write('Remove stopwords: '+ str(remove_stop_words) + '\n' )
  file.write('Remove ponctuation: '+ str(remove_ponctuation) + '\n' )
  file.write('\nAcuracia: ' + str(np.mean(predicted == test_target)) )
  file.write('\n')
  file.write(metrics.classification_report(test_target, predicted,target_names=categories))
  file.write('\n')
  file.write('Matriz de Confusao: \n\n')
  file.write(np.array2string(metrics.confusion_matrix(test_target, predicted), separator=', '))
  file.write('\n\n#############################################\n\n')
  file.close() 