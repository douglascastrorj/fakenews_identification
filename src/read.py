#!/usr/bin/env python
# -*- coding: utf-8 -*-

import read_dataset as rd

train,test = rd.read()
dataset = rd.get_text(train)

file = open('news.txt', 'a')

for text in dataset:
    file.write(text + '\n' )