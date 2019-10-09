#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate, ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import recall_score, confusion_matrix

import read_dataset as rd
from text_processor import Preprocessor
from features_combinations import get_combinations, use_custom


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
  sentiment=False,
  tag_ngram=False,
  text_features=False,
  file_path='log_politics_and_buzzfeed.txt'
):

  print('Features utilizadas: \n' )
  print('NGRAM: '+ str(n_gram) + '\n' )
  print('TAG_NGRAM: '+ str(tag_ngram) + '\n' )
  print('tags: '+ str(tags) + '\n' )
  print('pos: '+ str(pos) + '\n' )
  print('dep: '+ str(dep) + '\n' )
  print('stem: '+ str(stem) + '\n' )
  print('ent: '+ str(ent) + '\n' )
  print('alpha: '+ str(alpha) + '\n' )
  print('Remove stopwords: '+ str(remove_stop_words) + '\n' )
  print('Remove ponctuation: '+ str(remove_punct) + '\n' )
  print('lex: '+ str(lex) + '\n' )
  print('sentiment: '+ str(sentiment) + '\n' )
  print('text_features: '+ str(text_features) + '\n' )

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
                  lex=lex,
                  sentiment=sentiment,
                  tag_ngram=tag_ngram,
                  text_features=text_features
                )

  ##              TREINANDO NAIVE               ##
  print(train_text.shape)
  print(train_text[0])
  # 117670 total features
  # svd = TruncatedSVD(n_components=200, random_state=42)
  # train_text = svd.fit_transform(train_text)
  print ('Treinando modelo...')
  # text_clf = RandomForestClassifier()
  # text_clf = SVC(kernel='rbf', C=100, gamma=0.0001)
  # text_clf =  SGDClassifier(loss='hinge', penalty='l2',
  #                           alpha=1e-3, random_state=42,
  #                           max_iter=7, tol=None)
  text_clf = Pipeline([
                      # ('vect', CountVectorizer()),
                        # ('tfidf', TfidfTransformer()),
                        #('vect',TfidfVectorizer( ngram_range=(1, n_gram), max_df=0.5, min_df=2 )),
                        ('clf', MultinomialNB()),
                        # ('clf', SGDClassifier(loss='hinge', penalty='l2',
                        #                       alpha=1e-3, random_state=42,
                        #                       max_iter=7, tol=None)),
                ])

  file = open(file_path, 'a')
  file.write('Features utilizadas: \n' )
  file.write('NGRAM: '+ str(n_gram) + '\n' )
  file.write('TAG_NGRAM: '+ str(tag_ngram) + '\n' )
  file.write('pos: '+ str(pos) + '\n' )
  file.write('dep: '+ str(dep) + '\n' )
  file.write('tags: '+ str(tags) + '\n' )
  file.write('stem: '+ str(stem) + '\n' )
  file.write('ent: '+ str(ent) + '\n' )
  file.write('alpha: '+ str(alpha) + '\n' )
  file.write('lex: '+ str(lex) + '\n' )
  file.write('sentiment: '+ str(sentiment) + '\n' )
  file.write('text_features: '+ str(text_features) + '\n' )
  file.write('Remove stopwords: '+ str(remove_stop_words) + '\n' )
  file.write('Remove ponctuation: '+ str(remove_punct) + '\n\n' )

  kf = KFold(n_splits=10)
  f1 = []
  precision =[]
  recall = []
  accuracy = []
  for train_index, test_index in kf.split(train_text):
    # print('Kfold train_index: ', train_index, '\ntest_index: ', test_index)

    X_train, X_test = extract_indexes(train_text, train_index), extract_indexes(train_text, test_index)
    y_train, y_test = extract_indexes(train_target, train_index), extract_indexes(train_target, test_index)

    print(' train target ',extract_indexes(train_target, train_index))
    print(' test target ',extract_indexes(train_target, test_index))
    text_clf.fit(X_train, y_train)
    y_pred = text_clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print (metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=categories))

    file.write( metrics.classification_report(y_test, y_pred, target_names=categories) )

    precision.append(metrics.precision_score(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred))
    f1.append(metrics.f1_score(y_test, y_pred))
    accuracy.append(metrics.accuracy_score(y_test, y_pred))

  f1 = np.array(f1)
  precision = np.array(precision)
  recall = np.array(recall)
  accuracy = np.array(accuracy)

  f1_mean =f1.mean()
  precision_mean = precision.mean()
  recall_mean = recall.mean()
  accuracy_mean = accuracy.mean()

  f1_std = f1.std()
  precision_std = precision.std()
  recall_std = recall.std()
  accuracy_std = accuracy.std()

  print('Escrevendo arquivo de log\n')
  file.write('Recall Macro: ' + str(recall_mean) + ' (+/-) ' + str(recall_std * 2) + '\n' )
  file.write('Precision Macro: ' + str(precision_mean) + ' (+/-) ' + str(precision_std * 2) + '\n' )
  file.write('F1 Macro: ' + str(f1_mean) + ' (+/-) ' +str(f1_std * 2) + '\n' )
  file.write('Accuracy: ' + str(accuracy_mean) + ' (+/-) ' +str(accuracy_std * 2) + '\n' )

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
# combinations = use_custom()


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
                    lex=combination['lex'],
                    sentiment=combination['sentiment'],
                    tag_ngram=combination['tag_ngram'],
                    text_features=combination['text_features'],
                    file_path='log_text_features.txt'
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

