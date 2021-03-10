from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import numpy as np

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
    return X_train, X_test, y_train, y_test

def preprocess(X, y, vocabulary_size, input_length):
    data = X.copy()
    data.reset_index(inplace=True)

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

    for word in sentences:
        print(word)
        print("---")

    onehot_repr = [one_hot(words, vocabulary_size) for words in sentences]

    embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen= input_length) #  ensure that all sequences in a list have the same length
    X = np.array(embedded_docs)
    y = np.array(y)

    return split_data(X, y)