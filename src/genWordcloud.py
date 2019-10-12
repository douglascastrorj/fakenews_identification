#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from os import path
from wordcloud import WordCloud

from analise_dataset import getAdjetivos, getEnts, getDataset

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

def arrayToText(array):
    return ' '.join(array)


def generateWordCloud(text):
    # Read the whole text.
    # text = open(path.join(d, 'trustnews.txt'), encoding='utf8').read()

    # Generate a word cloud image
    wordcloud = WordCloud().generate(text)

    # Display the generated image:
    # the matplotlib way:
    import matplotlib.pyplot as plt
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40, background_color="white").generate(text)
    # plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


dataset_fake, dataset_real = getDataset(remove_stop = True)

adjFakenews = getAdjetivos(dataset_fake)
adjReal = getAdjetivos(dataset_real)

entsFake = getEnts(dataset_fake, entType='GPE')
entsReal = getEnts(dataset_real, entType='GPE')

print('Adjetivos Fake', len(adjFakenews))
print('Adjetivos Real', len(adjReal))

print('Ents Fake', len(entsFake))
print('Ents Real', len(entsReal))

# generateWordCloud(arrayToText(adjFakenews))
# generateWordCloud(arrayToText(adjReal))

generateWordCloud(arrayToText(entsFake))
generateWordCloud(arrayToText(entsReal))
