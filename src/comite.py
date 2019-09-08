#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate, ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, confusion_matrix

import read_dataset as rd
from text_processor import Preprocessor
from features_combinations import get_combinations

### PROCESSANDO TEXTO ######

def extract_indexes(arr, indexes):
  selected = []
  for index in indexes:
    selected.append(arr[index])
  return np.array(selected)

def analisar_features(
  train_text,
  n_gram = 1,
  pos=False,
  tags=False,
  dep=False,
  stem = False,
  remove_stop_words = False,
  remove_punct = False,
  ent=False,
  alpha=False,
  lex=False,
  file_path='log.txt'
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
                  alpha=alpha,
                  vectorizer='count',
                  lex=lex
                )

  ##              TREINANDO NAIVE               ##

  print ('Treinando modelo...')
  clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                      random_state=1)
  clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
  # clf3 = SGDClassifier(loss='hinge', penalty='l2',
  #                       alpha=1e-3, random_state=42,
  #                       max_iter=7, tol=None)
  clf3 = MultinomialNB()
  clf4 = SVC(C=100, gamma=5e-05, kernel='rbf')
  text_clf = VotingClassifier(estimators=[
               ('lr', clf1), ('rf', clf2), ('mnb', clf3), ('svm', clf4)], voting='hard')
 

  file = open(file_path, 'a')
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

  kf = KFold(n_splits=10)
  f1 = []
  precision =[]
  recall = []
  for train_index, test_index in kf.split(train_text):
    # print('Kfold train_index: ', train_index, '\ntest_index: ', test_index)

    X_train, X_test = extract_indexes(train_text, train_index), extract_indexes(train_text, test_index)
    y_train, y_test = extract_indexes(train_target, train_index), extract_indexes(train_target, test_index)

    print(' train target ',extract_indexes(train_target, train_index))
    print(' test target ',extract_indexes(train_target, test_index))
    text_clf.fit(X_train, y_train)
    y_pred = text_clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=categories))

    file.write( metrics.classification_report(y_test, y_pred, target_names=categories) )

    precision.append(metrics.precision_score(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred))
    f1.append(metrics.f1_score(y_test, y_pred))

  f1 = np.array(f1)
  precision = np.array(precision)
  recall = np.array(recall)

  f1_mean =f1.mean()
  precision_mean = precision.mean()
  recall_mean = recall.mean()

  f1_std = f1.std()
  precision_std = precision.std()
  recall_std = recall.std()

  print('Escrevendo arquivo de log\n')
  file.write('Recall Macro: ' + str(recall_mean) + ' (+/-) ' + str(recall_std * 2) + '\n' )
  file.write('Precision Macro: ' + str(precision_mean) + ' (+/-) ' + str(precision_std * 2) + '\n' )
  file.write('F1 Macro: ' + str(f1_mean) + ' (+/-) ' +str(f1_std * 2) + '\n' )

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

combinations = get_combinations()


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
                    ent=combination['ent'],
                    lex=False,
                    file_path='kfold_comite_4.txt'
                    )