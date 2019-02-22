#!/usr/bin/env python
# -*- coding: utf-8 -*-

def read(percent_train = .8):
    import os
    import json
    import random

    # folder_fake = 'FakeNewsNet-master/Data/BuzzFeed/FakeNewsContent/'
    # folder_real = 'FakeNewsNet-master/Data/BuzzFeed/RealNewsContent/'

    folder_fake = 'full_texts\\fake\\'
    folder_real = 'full_texts\\true\\'

    lista_fake = os.listdir(folder_fake)
    paths_fake =  [ folder_fake + file_name for file_name in lista_fake ]
    fake_dataset = []
    for path in paths_fake:
        with open(path) as f:
            data = f.read() #json.load(f)
            # print data.decode('latin-1').encode('utf-8')
            relevant_data = {
                                'text': data.decode('latin-1').encode('utf-8'),
                                'target': 0,
                                'target_name': 'fake'
                            }
            fake_dataset.append(relevant_data)

    lista_real = os.listdir(folder_real)
    paths_real =  [ folder_real + file_name for file_name in lista_real ]
    real_dataset = []
    for path in paths_real:
        with open(path) as f:
            data = f.read() #json.load(f)
            relevant_data = {
                                'text': data.decode('latin-1').encode('utf-8'),
                                'target': 1,
                                'target_name': 'real'
                            }
            real_dataset.append(relevant_data)

    fake_qtd = int(len(fake_dataset) * percent_train)
    real_qtd = int(len(real_dataset) * percent_train)

    random.shuffle(fake_dataset)
    random.shuffle(real_dataset)
    train = fake_dataset[:fake_qtd] + real_dataset[:real_qtd]
    test = fake_dataset[fake_qtd:] + real_dataset[real_qtd:]

    return train, test

    # dataset = fake_dataset + real_dataset    
    # random.shuffle(dataset)

    # train_qtd = int(len(dataset) * percent_train)
    # print train_qtd
    # return dataset[:train_qtd], dataset[train_qtd:]
def get_text(relevant_data):
    return [data['text'] for data in relevant_data]

def get_target(relevant_data):
    return [data['target'] for data in relevant_data]
    

# train, test = read()
# print len(train), len(test), train[1]
