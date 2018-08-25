#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian',
               'comp.graphics', 'sci.med']
#carregar dataset
twenty_train = fetch_20newsgroups(subset='train',
                categories=categories, shuffle=True, random_state=42)
print("\n".join(twenty_train.data[0].split("\n")[:3]))

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])


#Tokenizar texto
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
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


#Treinando Classificador
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

#construindo Pipeline
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
])

#treinando modelo
text_clf.fit(twenty_train.data, twenty_train.target)

#avaliacao de desempenho no conjunto de teste
import numpy as np
twenty_test = fetch_20newsgroups(subset='test',
     categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target) )


#treinando com svm
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)),
 ])
text_clf.fit(twenty_train.data, twenty_train.target)  

predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

#metricas
from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
        target_names=twenty_test.target_names))

print(metrics.confusion_matrix(twenty_test.target, predicted))































