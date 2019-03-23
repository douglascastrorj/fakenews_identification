from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import read_dataset as rd
from text_processor import Preprocessor


def get_features(corpus):

    vectorizer = CountVectorizer() 

    x = vectorizer.fit_transform(corpus)

    bag_of_words = x.toarray()
    dictionary = vectorizer.get_feature_names()

    print np.array(bag_of_words).shape
    print 'vocabulary has ', len(dictionary), ' words.'

    return bag_of_words, dictionary
    


def getWordsFrequency(bag_of_words, dictionary, howMany = 10):
    frequencies = np.zeros(len(dictionary))
    for i in range(0, len(bag_of_words) ):
        for j in range(0, len(bag_of_words[i]) ):
            frequencies[j] = frequencies[j] + bag_of_words[i][j]
    
    data = []
    for i in range(0, len(frequencies)):
        data.append({
            'word': dictionary[i],
            'frequency': frequencies[i]
        })
    return data

def getMostFrequentWords(words, howMany = 10):
    return sorted(words, key=lambda word: word['frequency'])[len(words)-1 - howMany: len(words)-1]

def writeFile(words, title='file.txt'):
    f= open(title,"w+")
    f.write('[')
    for word in mostFrequent:
        obj = '\n\t{\n\t\t"text": "' + str(word['word']) + '",\n\t\t"frequency" : ' + str(word['frequency']) + '\n\t},'
        f.write(obj)
    f.write(']')
    f.close()

def save_dictionary(dic, title):
    string = ''
    for key,value in dic.items():
        string = string + key + "\t"+ str(value) +'\n'
    print string
    f= open(title,"w+")
    f.write(string)
    f.close()

##################         MAIN          ##################
folders_fake = ['FakeNewsNet-master/Data/BuzzFeed/FakeNewsContent/', 'FakeNewsNet-master/Data/PolitiFact/FakeNewsContent/']
folders_real = ['FakeNewsNet-master/Data/BuzzFeed/RealNewsContent/', 'FakeNewsNet-master/Data/PolitiFact/RealNewsContent/']

fake_dataset = []
for folder_fake in folders_fake:
    dataset = rd.read_folder(folder_fake,0,'fake')
    fake_dataset =  fake_dataset + dataset

real_dataset = []
for folder_real in folders_real:
    dataset = rd.read_folder(folder_real,1,'real')
    real_dataset = real_dataset + dataset

corpus_fake = [ data['text'] for data in fake_dataset ]
corpus_real = [ data['text'] for data in real_dataset ]

corpus = []
answer = ''
while True:
    answer = raw_input('Calculate metrics for which dataset? (fake, real) ')
    if answer == 'fake':
        corpus = corpus_fake
        break
    elif answer == 'real':
        corpus = corpus_real
        break


corpus = corpus_real
## METRICS
bag, dic = get_features(corpus)

words = getWordsFrequency(bag, dic)
mostFrequent = getMostFrequentWords(words, 1000)

writeFile(mostFrequent, 'most_frequent_' + answer + '.json')

processor = Preprocessor()

# tokens = processor.tokenize(corpus[0])

from nltk.tag import pos_tag_sents, pos_tag

tags = pos_tag(corpus[0].strip().split(" "))

tags_count = {}

for tag in tags:
    try:
        key = tag[1]
        tags_count[key] = tags_count[key] + 1
    except:
        key = tag[1]
        tags_count[key] = 0
    

save_dictionary( tags_count, 'tags_frequency' + answer + '.csv')