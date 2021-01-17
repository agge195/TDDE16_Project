from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LassoCV
from yellowbrick.regressor import AlphaSelection
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
from keras.optimizers import Adam
from hyperas import optim
from hyperas.distributions import choice, uniform
import xgboost as xgb
import matplotlib.pyplot as plt
import re
import nltk
import pandas as pd
import numpy as np
import sys

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
    return X_train, X_test, y_train, y_test

def preprocess(X, y, vocabulary_size, sent_len):
    data = X.copy()
    data.reset_index(inplace=True)

    # Stemming since word meaning isn't that important in this context.
    porter_stemmer = PorterStemmer()
    sentences = []

    # preprocesses the text used in training,
    for i in range(0, len(data)):
        review = re.sub('[^a-zA-Z]', ' ', data['title'][i])
        review = review.lower()
        review = review.split()

        # do stemming if word isn't a stop word (such as "the", "a", "an")
        review = [porter_stemmer.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        sentences.append(review)

    onehot_repr = [one_hot(words, vocabulary_size) for words in sentences]

    embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen= sent_len) #  ensure that all sequences in a list have the same length
    X = np.array(embedded_docs)
    y = np.array(y)

    return split_data(X, y)


def tune_LSTM(params):
    max_features = 40

    model = Sequential()
    model.add(Embedding(vocabulary_size, max_features, input_length=sent_len))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(units=params['unit_1'], return_sequences=True))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(units=params['unit_2'], return_sequences=False))
    model.add(Dropout(params['dropout']))
    model.add(Dense(1, activation=params['activation']))

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=params['learning_rate']),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

    pred_auc = model.predict_proba(X_test, batch_size=64, verbose=0)

    # acc = roc_auc_score(y_test, pred_auc)

    pred = model.predict_classes(X_test)
    acc = accuracy_score(y_test, pred)
    print(acc)

    # print('AUC:', acc)
    sys.stdout.flush()

    return {'loss': 1
        -acc, 'status': STATUS_OK}

def build_LSTM(vocabulary_size, sent_len, tune=False):
    max_features = 40  # vector features

    if tune:
        # Current best (using AUC scoring): activation: sigmoid, dropout: 0.25, learning rate: 0.005, unit_1: 100, unit_2: 80
        # no1 best (AUC scoring): activation: sigmoid, dropout: 0.6, learning rate: 0.0005, unit_1: 100, unit_2: 100
        # best with accuracy scoring: activation: sigmoid, dropout: 0.6, learning rate: 0.0005, unit_1: 100, unit_2: 64
            params = {
                     'unit_1': hp.choice('unit_1', [100, 120]),
                     'unit_2': hp.choice('unit_2', [100, 80, 64]),
                     'dropout': hp.choice('dropout', [0.3, 0.6, 0.25]),
                     'learning_rate': hp.choice('learning_rate', [0.0005, 0.001, 0.005, 0.01]),
                     'activation': hp.choice('activation', ['relu', 'sigmoid'])
                     }

            trials = Trials()
            res_best = fmin(fn=tune_LSTM,
                            space=params,
                            algo=tpe.suggest,
                            max_evals=10,
                            trials=trials)

            print("best params: {}".format(res_best))
            return 0

    else:
        model = Sequential()
        model.add(Embedding(vocabulary_size, max_features, input_length= sent_len))
        model.add(Dropout(0.3))
        model.add(LSTM(100, return_sequences= True))
        model.add(Dropout(0.3))
        model.add(LSTM(64, return_sequences= False))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer= Adam(learning_rate= 0.0005), metrics=['accuracy'])

    return model

def build_bidirectional_LSTM(vocabulary_size, sent_len):
    max_features = 40 # vector features, **
    model = Sequential()
    model.add(Embedding(vocabulary_size, max_features, input_length= sent_len))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer= Adam(learning_rate= 0.0005), metrics=['accuracy'])
    return model

def do_LSTM(model, X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=30,batch_size=64)
    pred = model.predict_classes(X_test)
    acc = accuracy_score(y_test, pred)
    print(acc)
    print(classification_report(y_test, pred))

    pred = pred.ravel()

    return pred

# Not used
def do_lasso(X_train, X_test, y_train, y_test, cv=True):
    #model = Lasso(alpha=0.1)
    alpha = 0.133
    if cv:
        alphas = np.logspace(-10, 1, 400)
        model = LassoCV(alphas=alphas)
        visualizer = AlphaSelection(model)
        visualizer.fit(X_train, y_train)
        visualizer.show()  # optimal is found from a logspace of 400 possible alphas
        alpha = visualizer.estimator.alpha_

    model = Lasso(alpha)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = mean_squared_error(y_test, pred)

    print("mse from y_yest: {}".format(score))

    return pred
#

def do_Multinomial_NaiveBayes(X_train, y_train, X_test, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("Accuracy: {}".format(acc))
    print(classification_report(y_test, pred))

    return pred

def tune_XGBoost(params):
    model = xgb.XGBClassifier(max_depth=params['max_depth'],
                              min_child_weight=params['learning_rate'],
                              subsample=params['subsample'],
                              gamma=params['gamma'],
                              colsample_bytree=params['colsample_bytree'],
                              n_estimators= params['n_estimators'],
                              learning_rate=params['learning_rate']
                              )

    model.fit(X_train, y_train)
    accs = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
    CV_mean = accs.mean()
    print("CV mean: {}".format(CV_mean))

    return {'loss':1-CV_mean, 'status': STATUS_OK}


def do_XGBoost(X_train, y_train, X_test, y_test, tune=False):

    if tune:
        params = {
            'max_depth': hp.choice('max_depth', range(5, 30, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.0001, 0.5),
            'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
            'gamma': hp.quniform('gamma', 0, 0.50, 0.01),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)
        }

        trials = Trials()
        res_best = fmin(fn=tune_XGBoost,
                        space=params,
                        algo=tpe.suggest,
                        max_evals=50,
                        trials=trials)

        print("best params: {}".format(res_best))
        model = xgb.XGBClassifier(max_depth=res_best['max_depth'],
                                  learning_rate=res_best['learning_rate'],
                                  gamma= res_best['gamma'],
                                  min_child_weight=res_best['min_child_weight'],
                                  subsample=res_best['subsample'],
                                  colsample_bytree=res_best['colsample_bytree'])

        model.fit(X_train, y_train)

    else:
        # Current best parameters
        model = xgb.XGBClassifier(max_depth=24,
                                  learning_rate=0.07,
                                  gamma=0.16,
                                  min_child_weight=7.0,
                                  subsample=0.65,
                                  colsample_bytree=0.18)
        model.fit(X_train, y_train)

    pred = model.predict(X_test)
    pred = [round(value) for value in pred]
    print(classification_report(y_test, pred))

    # best params: {'colsample_bytree': 0.18, 'gamma': 0.16, 'learning_rate': 0.07, 'max_depth': 24, 'min_child_weight': 7.0, 'n_estimators': 29, 'subsample': 0.65}
    return pred

def do_all_plot_roc(X_train, X_test, y_train, y_test):
    model_LSTM = build_LSTM(vocabulary_size, sent_len)
    model_bidirectional_LSTM = build_bidirectional_LSTM(vocabulary_size, sent_len)

    pred_LSTM = do_LSTM(model_LSTM, X_train, X_test, y_train, y_test)
    pred_bidirectional_LSTM = do_LSTM(model_bidirectional_LSTM, X_train, X_test, y_train, y_test)
    pred_MNB = do_Multinomial_NaiveBayes(X_train, y_train, X_test, y_test)
    pred_XGB = do_XGBoost(X_train, y_train, X_test, y_test, tune=False)

    fposr_LSTM, tposr_LSTM, thresh_LSTM = roc_curve(y_test, pred_LSTM)
    auc_LSTM = auc(fposr_LSTM, tposr_LSTM)

    fposr_BI_LSTM, tposr_BI_LSTM, thresh_BI_LSTM = roc_curve(y_test, pred_bidirectional_LSTM)
    auc_BI_LSTM = auc(fposr_BI_LSTM, tposr_BI_LSTM)

    fposr_MNB, tposr_MNB, thresh_MNB = roc_curve(y_test, pred_MNB)
    auc_MNB = auc(fposr_MNB, tposr_MNB)

    fposr_XGB, tposr_XGB, thresh_XGB = roc_curve(y_test, pred_XGB)
    auc_XGB = auc(fposr_XGB, tposr_XGB)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fposr_LSTM, tposr_LSTM, label='LSTM, AUC = {:.3f}'.format(auc_LSTM))
    plt.plot(fposr_BI_LSTM, tposr_BI_LSTM, label='Bidirectional LSTM, AUC = {:.3f}'.format(auc_BI_LSTM))
    plt.plot(fposr_MNB, tposr_MNB, label='Multinomial Naive Bayes, AUC = {:.3f}'.format(auc_MNB))
    plt.plot(fposr_XGB, tposr_XGB, label='XGBoost, AUC = {:.3f}'.format(auc_XGB))


    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fposr_LSTM, tposr_LSTM, label='LSTM, AUC = {:.3f}'.format(auc_LSTM))
    plt.plot(fposr_BI_LSTM, tposr_BI_LSTM, label='Bidirectional LSTM, AUC = {:.3f}'.format(auc_BI_LSTM))
    plt.plot(fposr_MNB, tposr_MNB, label='Multinomial Naive Bayes, AUC = {:.3f}'.format(auc_MNB))
    plt.plot(fposr_XGB, tposr_XGB, label='XGBoost, AUC = {:.3f}'.format(auc_XGB))

    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.title('ROC curve, zoomed in')
    plt.legend(loc='best')
    plt.show()

# label = 0: reliable news
# label = 1: unreliable news
input_data = pd.read_csv('input_data/train.csv')

# drop NA values
input_data = input_data.dropna()

# remove target variable from dataset for X
X = input_data.drop('label', axis=1)

# y has the target variable
y = input_data['label']
vocabulary_size = 8000
sent_len = 20
#nltk.download('stopwords')

X_train, X_test, y_train, y_test = preprocess(X, y, vocabulary_size, sent_len)
#model = build_LSTM(vocabulary_size, sent_len, tune=False)
#model = build_bidirectional_LSTM(vocabulary_size, sent_len)
#do_LSTM(model, X_train, X_test, y_train, y_test)
#do_lasso(X_train, X_test, y_train, y_test)
#do_Multinomial_NaiveBayes(X_train, y_train, X_test, y_test)
#do_all_plot_roc(X_train, X_test, y_train, y_test)
do_XGBoost(X_train, y_train, X_test, y_test, tune=True)

# TODO: Visualize the results better (Done?)