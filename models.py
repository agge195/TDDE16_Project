from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
from keras.optimizers import Adam
import xgboost as xgb
import matplotlib.pyplot as plt
import sys

vocabulary_size = 8000
input_length = 20

def tune_LSTM(params):
    max_features = 40
    X_train = params['X_train']
    X_test = params['X_test']
    y_train = params['y_train']
    y_test = params['y_test']

    model = Sequential()
    model.add(Embedding(vocabulary_size, max_features, input_length=input_length))
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

def build_LSTM(vocabulary_size, input_length, X_train, y_train, X_test, y_test, tune=False):
    max_features = 40  # vector features

    if tune:
        # Current best (using AUC scoring): activation: sigmoid, dropout: 0.25, learning rate: 0.005, unit_1: 100, unit_2: 80
        # no1 best (AUC scoring): activation: sigmoid, dropout: 0.6, learning rate: 0.0005, unit_1: 100, unit_2: 100
        # best with accuracy scoring: activation: sigmoid, dropout: 0.6, learning rate: 0.0005, unit_1: 100, unit_2: 64
            params = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
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
        model.add(Embedding(vocabulary_size, max_features, input_length= input_length))
        model.add(Dropout(0.6))
        model.add(LSTM(100, return_sequences= True))
        model.add(Dropout(0.6))
        model.add(LSTM(64, return_sequences= False))
        model.add(Dropout(0.6))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer= Adam(learning_rate= 0.0001), metrics=['accuracy'])

    return model

def build_bidirectional_LSTM(vocabulary_size, input_length):
    max_features = 40
    model = Sequential()
    model.add(Embedding(vocabulary_size, max_features, input_length= input_length))
    model.add(Dropout(0.6))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.6))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer= Adam(learning_rate= 0.0005), metrics=['accuracy'])
    return model

def do_LSTM(model, X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5,batch_size=64)
    pred = model.predict_classes(X_test)
    acc = accuracy_score(y_test, pred)
    print(acc)
    print(classification_report(y_test, pred))
    pred = pred.ravel()

    fposr_LSTM, tposr_LSTM, thresh_LSTM = roc_curve(y_test, pred)
    auc_LSTM = auc(fposr_LSTM, tposr_LSTM)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fposr_LSTM, tposr_LSTM, label='LSTM, AUC = {:.3f}'.format(auc_LSTM))
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    return pred

def do_Multinomial_NaiveBayes(X_train, y_train, X_test, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("Accuracy: {}".format(acc))
    print(classification_report(y_test, pred))

    return pred

def tune_XGBoost(params):
    X_train = params['X_train']
    y_train = params['y_train']

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
            'X_train': X_train,
            'y_train': y_train,
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
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)


    pred = model.predict(X_test)
    pred = [round(value) for value in pred]
    print(classification_report(y_test, pred))
    fposr_XGB, tposr_XGB, thresh_XGB = roc_curve(y_test, pred)
    auc_XGB = auc(fposr_XGB, tposr_XGB)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fposr_XGB, tposr_XGB, label='XGBoost, AUC = {:.3f}'.format(auc_XGB))
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    # best params: {'colsample_bytree': 0.18, 'gamma': 0.16, 'learning_rate': 0.07, 'max_depth': 24, 'min_child_weight': 7.0, 'n_estimators': 29, 'subsample': 0.65}
    # other best params: {'colsample_bytree': 0.41000000000000003, 'gamma': 0.23, 'learning_rate': 0.03684872782247344, 'max_depth': 6, 'min_child_weight': 9.0, 'n_estimators': 28, 'subsample': 0.72}
    return pred
