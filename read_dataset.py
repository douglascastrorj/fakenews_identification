#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random

def read_folder(folder_path, target=0, target_name='fake'):
    import os
    import json

    dataset = []
    files = os.listdir(folder_path)
    files_path =  [ folder_path + file_name for file_name in files ]
    
    for path in files_path:
        with open(path, 'r') as f:
            obj = json.load(f)
            data = obj['text']
            # print data.decode('latin-1').encode('utf-8')
            relevant_data = {
                                'text': data,
                                'target': target,
                                'target_name': target_name
                            }
            dataset.append(relevant_data)
    return dataset

def read(percent_train = 1):

    folders_fake = ['FakeNewsNet-master/Data/BuzzFeed/FakeNewsContent/', 'FakeNewsNet-master/Data/PolitiFact/FakeNewsContent/']
    folders_real = ['FakeNewsNet-master/Data/BuzzFeed/RealNewsContent/', 'FakeNewsNet-master/Data/PolitiFact/RealNewsContent/']

    fake_dataset = []
    for folder_fake in folders_fake:
        dataset = read_folder(folder_fake,0,'fake')
        fake_dataset =  fake_dataset + dataset

    real_dataset = []
    for folder_real in folders_real:
        dataset = read_folder(folder_real,1,'real')
        real_dataset = real_dataset + dataset

    fake_qtd = int(len(fake_dataset) * percent_train)
    real_qtd = int(len(real_dataset) * percent_train)

    random.shuffle(fake_dataset)
    random.shuffle(real_dataset)
    train = fake_dataset[:fake_qtd] + real_dataset[:real_qtd]
    test = fake_dataset[fake_qtd:] + real_dataset[real_qtd:]

    return train, test

def get_text(relevant_data):
    return [data['text'] for data in relevant_data]

def get_target(relevant_data):
    return [data['target'] for data in relevant_data]
    

# train, test = read()
# print len(train), len(test), train[1]
