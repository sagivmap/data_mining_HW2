path_to_corpus_folder = 'donotdelete'
train_folder = 'training'
test_folder = 'test'

# Tokenization & Normalization

import os
import nltk
#nltk.download() #download all packages for the first time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

print("Start to clean data")
# lemmatizer = WordNetLemmatizer()
# ps = PorterStemmer()
# common_word_to_ignore = ['p', 'le', 'patient', 'study']
# for root, dirs, files in os.walk(path_to_corpus_folder):
#     for name in files:
#         with open(os.path.join(root, name), 'r+') as f:
#             data = f.read()
#             # lower case
#             data = data.lower()
#             # tokenization
#             tokens = word_tokenize(data)
#             # Remove punctuation tokens
#             tokens = [word for word in tokens if word.isalpha()]
#             # remove stop words
#             stop_words = set(stopwords.words('english'))
#             filtered_sentence = [w for w in tokens if not w in stop_words]
#             #Lemmatization
#             lemmatizered_sentence = []
#             for word in filtered_sentence:
#                 lemmatized_word = lemmatizer.lemmatize(word)
#                 if lemmatized_word not in common_word_to_ignore:
#                     lemmatizered_sentence.append(lemmatized_word)
#             data = ' '.join(lemmatizered_sentence)
#
#             f.seek(0)
#             f.write(data)
#             f.truncate()
#             f.close()
print("Finished to clean data")
print ("Start to count the words")
# categories = {}
# for root, dirs, files in os.walk(path_to_corpus_folder):
#     index = 0
#     ans = {}
#
#     for name in files:
#         with open(os.path.join(root, name), 'r+') as f:
#             data = f.read()
#             data = data.split(' ')
#
#             for word in data:
#                 if word in ans:
#                     ans[word] = ans[word] + 1
#                 else:
#                     ans[word] = 1
#         index += 1
#     if 'training' in root and root[-7:] != 'raining':
#         categories.update({root[-3:]: index})
#         from collections import defaultdict
#
#         dct = defaultdict(list)
#
#         for k, v in ans.items():
#             dct[v].append(k)
#         print('Top 10 words in ' + root[-3:])
#
#         from prettytable import PrettyTable
#
#         t = PrettyTable(['Word', '# of apps'])
#         for k, v in sorted(dct.items(), reverse=True)[:10]:
#             t.add_row([', '.join(v), k])
#
#         #print the table
#         print(t)
#
#
# for key, value in categories.items():
#     print('category: ' + key + ', # of docs: ' + str(value))

print ("Finished to count the words")

print ("START TASK 2")
# Loading the data sets from disk
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
data_train = load_files(os.path.join(path_to_corpus_folder,train_folder))
data_test = load_files(os.path.join(path_to_corpus_folder,test_folder))

def size_mb(docs):
    return sum(len(s.decode('utf-8')) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

vectorizer_TF_IDF = TfidfVectorizer()
vectorizer_Counter = CountVectorizer()
print ("Start to Feature Extraction with TfidfVectorizer\n")
from time import time
t0 = time()
X_train_TF_IDF = vectorizer_TF_IDF.fit_transform(data_train.data)
duration_to_TF_IDF_FE_train_data = time() - t0

print("TrainData:\ndone in %fs at %0.3fMB/s" %
      (duration_to_TF_IDF_FE_train_data, data_train_size_mb / duration_to_TF_IDF_FE_train_data))
print("n_samples: %d, n_features: %d" % X_train_TF_IDF.shape)
print()

t0 = time()
X_test_TF_IDF = vectorizer_TF_IDF.transform(data_test.data)
duration_to_TF_IDF_FE_test_data = time() - t0
print("TestData:\ndone in %fs at %0.3fMB/s" %
      (duration_to_TF_IDF_FE_test_data, data_test_size_mb / duration_to_TF_IDF_FE_test_data))
print("n_samples: %d, n_features: %d" % X_test_TF_IDF.shape)
print('\n')

print ("Start to Feature Extraction with CountVectorizer\n")
t0 = time()
X_train_Counter = vectorizer_Counter.fit_transform(data_train.data)
duration_to_Counter_FE_train_data = time() - t0

print("TrainData:\ndone in %fs at %0.3fMB/s" %
      (duration_to_Counter_FE_train_data, data_train_size_mb / duration_to_Counter_FE_train_data))
print("n_samples: %d, n_features: %d" % X_train_Counter.shape)
print()

t0 = time()
X_test_counter = vectorizer_Counter.transform(data_test.data)
duration_to_Counter_FE_test_data = time() - t0
print("TestData:\ndone in %fs at %0.3fMB/s" %
      (duration_to_Counter_FE_test_data, data_test_size_mb / duration_to_Counter_FE_test_data))
print("n_samples: %d, n_features: %d" % X_test_counter.shape)
print()

y_train, y_test = data_train.target, data_test.target

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def benchmark(clf, X_train, X_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


print ("START TO CLASIFY\n")

for X_train, X_test, FE_method in ((X_train_TF_IDF, X_test_TF_IDF, 'TF_IDF'),
                        (X_train_Counter, X_test_counter, 'Bag Of Words')):
    results = []
    for clf, name in (
            (SGDClassifier(), "SVM"),
            (MultinomialNB(), "Naive Bayes")):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf, X_train, X_test))
    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score with: " + FE_method)
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
             color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()
    print('=' * 80)









