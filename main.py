import pandas as pd
from models import *
from preprocessing import *

#nltk.download('stopwords')

# Retrain and combine plot
def do_all_plot_roc(X_train, X_test, y_train, y_test):
    model_LSTM = build_LSTM(vocabulary_size, input_length)
    model_bidirectional_LSTM = build_bidirectional_LSTM(vocabulary_size, input_length)

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
input_data = pd.read_csv('../input_data/train.csv')

# drop NA values
input_data = input_data.dropna()

# remove target variable from dataset for X
X = input_data.drop('label', axis=1)

# y has the target variable
y = input_data['label']

X_train, X_test, y_train, y_test = preprocess(X, y, vocabulary_size, input_length)
model = build_LSTM(vocabulary_size, input_length, X_train, y_train, X_test, y_test, tune=False)
# model = build_bidirectional_LSTM(vocabulary_size, input_length)
do_LSTM(model, X_train, X_test, y_train, y_test)
# do_lasso(X_train, X_test, y_train, y_test)
# do_Multinomial_NaiveBayes(X_train, y_train, X_test, y_test)
# do_all_plot_roc(X_train, X_test, y_train, y_test)
# do_XGBoost(X_train, y_train, X_test, y_test, tune=False)
print("")