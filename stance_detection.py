import read_dataset as rd
import random
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

def getDataset():
    folders_fake = ['FakeNewsNet-master/Data/BuzzFeed/FakeNewsContent/', 'FakeNewsNet-master/Data/PolitiFact/FakeNewsContent/']
    folders_real = ['FakeNewsNet-master/Data/BuzzFeed/RealNewsContent/', 'FakeNewsNet-master/Data/PolitiFact/RealNewsContent/']

    fake_buzz = rd.read_folder(folders_fake[0], target=0, target_name='BuzzFeed')
    real_buzz = rd.read_folder(folders_real[0], target=0, target_name='BuzzFeed')

    fake_politc = rd.read_folder(folders_fake[1], target=1, target_name='PolitiFact')
    real_politic = rd.read_folder(folders_real[1], target=1, target_name='PolitiFact')

    buzz_dataset = fake_buzz + real_buzz
    politic_dataset = fake_politc + real_politic

    print len(buzz_dataset)
    print len(politic_dataset)

    return buzz_dataset, politic_dataset

def split_train_test(dataset, percent_train=0.7):
    random.shuffle(dataset)
    qtd_train = int(len(dataset) * percent_train)

    train = dataset[:qtd_train]
    test = dataset[qtd_train:]

    return train,test

print 'reading dataset...'
buzz, politic = getDataset()
buzz_train,buzz_test = split_train_test(buzz)
politic_train,politic_test = split_train_test(politic)

train = buzz_train + politic_train
test = buzz_test + politic_test

train_text = rd.get_text(train)
train_target = rd.get_target(train)
test_text = rd.get_text(test)
test_target = rd.get_target(test)

print ('Treinando modelo...')
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=7, tol=None)),
 ])
text_clf.fit(train_text , train_target)
print( 'Treino concluido.')

#avaliacao de desempenho no conjunto de teste
predicted = text_clf.predict(test_text)
print('acuracia ', np.mean(predicted == test_target) )


categories = ['buzz', 'politics']
targets = [ data['target'] for data in test ]

from sklearn import metrics
print(metrics.classification_report(targets, predicted,target_names=categories))

print(metrics.confusion_matrix(targets, predicted))
