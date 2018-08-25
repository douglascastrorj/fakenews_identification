#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

import read_dataset as rd

train,test = rd.read(percent_train=.9)

categories = ['fake', 'real']


#Tokenizar texto
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform( [data['text'] for data in train] )
print(X_train_counts.shape)


#ocorrencias para frequencias
#Occurrence count is a good start but there is an issue: longer documents will have higher average 
#count values than shorter documents, even though they might talk about the same topics.
#To avoid these potential discrepancies it suffices to divide the number of occurrences of each word in a document by the total number of words in the document: these new features are called tf for Term Frequencies.
#Another refinement on top of tf is to downscale weights for words that occur in many documents in the corpus and are therefore less informative than those that occur only in a smaller portion of the corpus.

from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print(X_train_tfidf.shape)


print ('Pipeline')
#construindo Pipeline
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
])

print ('Treinando modelo...')
# #treinando modelo
text_clf.fit([data['text'] for data in train], [data['target'] for data in train])
print( 'Treino concluido.')

# #avaliacao de desempenho no conjunto de teste
import numpy as np
docs_test = [ data['text'] for data in test ]
targets = [ data['target'] for data in test ]
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == targets) )


# #treinando com svm
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)),
 ])
text_clf.fit([data['text'] for data in train], [data['target'] for data in train])

predicted = text_clf.predict(docs_test)
print(np.mean(predicted == targets) )

# #metricas
from sklearn import metrics
print(metrics.classification_report(targets, predicted,target_names=categories))

print(metrics.confusion_matrix(targets, predicted))
