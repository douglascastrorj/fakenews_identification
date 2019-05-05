import read_dataset as rd
import text_processor as tp
import random
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

def getDataset():
    folders_fake = [ 'FakeNewsNet-master/Data/PolitiFact/FakeNewsContent/', 'FakeNewsNet-master/Data/BuzzFeed/FakeNewsContent/' ]
    folders_real = [ 'FakeNewsNet-master/Data/PolitiFact/RealNewsContent/', 'FakeNewsNet-master/Data/BuzzFeed/RealNewsContent/']

    politic = {
        'real': rd.read_folder(folders_real[0], target=1, target_name='real'),
        'fake': rd.read_folder(folders_fake[0], target=0, target_name='fake')
    }
    
    buzz = {
        'real': rd.read_folder(folders_real[1], target=1, target_name='real'),
        'fake': rd.read_folder(folders_fake[1], target=0, target_name='fake')
    }
    
    return politic, buzz


def split_train_test(dataset, percent_train=0.7):
    random.shuffle(dataset)
    qtd_train = int(len(dataset) * percent_train)

    train = dataset[:qtd_train]
    test = dataset[qtd_train:]

    return train,test

def getClassifier():
    return Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='log', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=16, tol=None)),
                    # ('clf', LogisticRegression(n_jobs=3, C=1e5)),
                    # ('clf', MultinomialNB()),
    ])

print 'reading dataset...'
politicDataset, buzzDataset = getDataset()

train_real_politic, test_real_politic = split_train_test(politicDataset['real'])
train_fake_politic, test_fake_politic = split_train_test(politicDataset['fake'])

train_real_buzz, test_real_buzz = split_train_test(buzzDataset['real'])
train_fake_buzz, test_fake_buzz = split_train_test(buzzDataset['fake'])


##criar classificador q classifica como fake ou real apenas para politics facts
print 'Criando classificador para Politics Facts'
train_text = rd.get_text(train_real_politic) + rd.get_text(train_fake_politic)
train_target = rd.get_target(train_real_politic) + rd.get_target(train_fake_politic)

test_text = rd.get_text(test_real_politic) + rd.get_text(test_fake_politic)
test_target = rd.get_target(test_real_politic) + rd.get_target(test_fake_politic)

politic_clf = getClassifier()
politic_clf.fit(train_text, train_target)

predicted = politic_clf.predict(test_text)
print 'Acuracia Politcs Classifier: ', np.mean(predicted == test_target)
print '\n\n'


#criar classificador generico
print ' Criando Classificador Generico'

train_text = rd.get_text(train_real_politic) + rd.get_text(train_fake_politic) + rd.get_text(train_real_buzz) + rd.get_text(train_fake_buzz)
train_target = rd.get_target(train_real_politic) + rd.get_target(train_fake_politic) + rd.get_target(train_real_buzz) + rd.get_target(train_fake_buzz)

test_text = rd.get_text(test_real_politic) + rd.get_text(test_fake_politic) + rd.get_text(test_real_buzz) + rd.get_text(test_fake_buzz)
test_target = rd.get_target(test_real_politic) + rd.get_target(test_fake_politic) + rd.get_target(test_real_buzz) + rd.get_target(test_fake_buzz)

clf = getClassifier()
clf.fit(train_text, train_target)

predicted = clf.predict(test_text)
print 'Acuracia General Classifier: ', np.mean(predicted == test_target)
print '\n\n'