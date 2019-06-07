import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import read_dataset as rd
from text_processor import Preprocessor

#Read text
train,test = rd.read()

train_text = rd.get_text(train)
y = rd.get_target(train)

#proccess text
processor = Preprocessor()
train_text =  processor.process_dataset(
                  train_text, 
                  n_gram=2, 
                  stem=True, 
                  tags=True,
                  remove_stop_words=True, 
                  remove_punct=True,
                  pos=True,
                  dep=True,
                  alpha=True
                )

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_text)

features = vectorizer.get_feature_names()

# print(train_text[0], y[0])

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
limit = 1000
has_more_than_limit = X.shape[1] > limit
max_features = limit if has_more_than_limit else X.shape[1]

file = open('./logs/feature_importances.txt', 'a')
for f in range(0, max_features):
    print("%d. feature %d-%s (%f)" % (f + 1, indices[f], features[indices[f]], importances[indices[f]]))
    file.write( features[indices[f]]+ "\t" +str(importances[indices[f]])+"\n")
file.close()
print(len(features))
# Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(X.shape[1]), indices)
# plt.xlim([-1, X.shape[1]])
# plt.show()