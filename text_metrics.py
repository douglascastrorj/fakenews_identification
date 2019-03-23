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

def writeFile(words):
    f= open("words.json","w+")
    f.write('[')
    for word in mostFrequent:
        obj = '\n\t{\n\t"text": "' + str(word['word']) + '",\n\t"frequency" : ' + str(word['frequency']) + '\n\t},'
        f.write(obj)
    f.write(']')
    f.close()

def print_dictionary(dic):
    for key,value in dic.items():
        print key, "\t", value


##################         MAIN          ##################

folder = 'FakeNewsNet-master/Data/BuzzFeed/RealNewsContent/'
corpus = [ data['text'] for data in rd.read_folder(folder) ]


bag, dic = get_features(corpus)

words = getWordsFrequency(bag, dic)
mostFrequent = getMostFrequentWords(words, 1000)

# for word in mostFrequent:
#     print 'Word: ',word['word'],'\t\t\tFrequency: ', word['frequency']

processor = Preprocessor()

# tokens = processor.tokenize(corpus[0])

from nltk.tag import pos_tag_sents, pos_tag

sentence = 'I am a good boy'
tags = pos_tag(corpus[0].strip().split(" "))

tags_count = {}

for tag in tags:
    try:
        key = tag[1]
        tags_count[key] = tags_count[key] + 1
    except:
        key = tag[1]
        tags_count[key] = 0
    

print_dictionary( tags_count)