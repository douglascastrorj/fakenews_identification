from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import read_dataset as rd
from text_processor import Preprocessor
train,test = rd.read()
categories = ['fake', 'real']

train_text = rd.get_text(train)
train_target = rd.get_target(train)

processor = Preprocessor()

X =  processor.process_dataset(
                  train_text, 
                  n_gram=2, 
                  stem=True, 
                  tags=False,
                  remove_stop_words=False, 
                  remove_punct=False,
                  pos=False,
                  dep=False,
                  alpha=False,
                  vectorizer='count',
                  lex=False
                )
y = train_target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Set the parameters by cross-validation from svm
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.5e-4, 0.5e-5, 1e-5, 0.5e-6, 1e-6],
#                      'C': [1000, 5000, 7000, 10000]},
#                 ]


#param for random forest
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

# scores = ['precision', 'recall']
scores = ['f1']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    # clf = GridSearchCV(SVC(), tuned_parameters, cv=10,
    #                    scoring='%s_macro' % score)
    rfc=RandomForestClassifier(random_state=42)
    clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
