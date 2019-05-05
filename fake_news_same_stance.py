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
    folders_fake = [ 'FakeNewsNet-master/Data/PolitiFact/FakeNewsContent/']
    folders_real = [ 'FakeNewsNet-master/Data/PolitiFact/RealNewsContent/']

    real = rd.read_folder(folders_real[0], target=0, target_name='real')
    fake = rd.read_folder(folders_fake[0], target=1, target_name='fake')
    
    train_real, test_real = split_train_test(real)
    train_fake, test_fake = split_train_test(fake)

    train = train_fake + train_real
    test = test_fake + test_real

    return train,test

def split_train_test(dataset, percent_train=0.7):
    random.shuffle(dataset)
    qtd_train = int(len(dataset) * percent_train)

    train = dataset[:qtd_train]
    test = dataset[qtd_train:]

    return train,test

print 'reading dataset...'
train, test = getDataset()

preprocessor = tp.Preprocessor()

train_text = rd.get_text(train)
train_target = rd.get_target(train)
test_text = rd.get_text(test)
test_target = rd.get_target(test)

train_text = [ preprocessor.n_gram(train_text[i], n=4) for i in range(0, len(train_text))]
test_text = [ preprocessor.n_gram(test_text[i], n=4) for i in range(0, len(test_text))]

print ('Treinando modelo...')
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='log', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=16, tol=None)),
                    # ('clf', LogisticRegression(n_jobs=3, C=1e5)),
                    # ('clf', MultinomialNB()),
 ])
text_clf.fit(train_text , train_target)
print( 'Treino concluido.')

#avaliacao de desempenho no conjunto de teste
predicted = text_clf.predict(test_text)
print('acuracia ', np.mean(predicted == test_target) )


categories = ['real', 'fake']
targets = [ data['target'] for data in test ]

from sklearn import metrics
print(metrics.classification_report(targets, predicted,target_names=categories))

print(metrics.confusion_matrix(targets, predicted))
