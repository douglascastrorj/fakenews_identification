"""
Author: Douglas Castro da Silva
Description: this file contains contains the implementations of functions to extract custom features from text
"""
import re

def extract_link(text):
    link_regex = r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
    tokens = text.split(' ')

    for i in range(0, len(tokens)):
        contains_link = re.search(link_regex, tokens[i])
        if contains_link:
            tokens[i] = '@feature_link'
    return ' '.join(tokens)

def many_exclamation_mark(text):
    return re.sub(r'\!+',  ' @feature_many_interrogation_mark ', text)

def many_interrogation_mark(text):
    return re.sub(r'\?+', ' @feature_many_interrogation_mark ', text)

def exclamation_interrogation(text):
    string = re.sub(r'((!\?)+|(\?!)+)(\?|!)+',  ' @feature_exclamation_interrogation ',   text)
    return string


# print( extract_link('some text with a link to http://somedomain.com and more text.'))
# print( many_exclamation_mark('oh my god!!! no fucking way!!!!!') )
# print( exclamation_interrogation('serius man?!?!!!') )