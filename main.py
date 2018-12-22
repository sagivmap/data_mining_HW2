path_to_corpus_folder = 'ohsumed-first-20000-docs'
train_folder = 'training'
test_folder = 'test'

# Tokenization & Normalization

import os
import nltk
#nltk.download()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
index = 0
for root, dirs, files in os.walk(path_to_corpus_folder):
    for name in files:
        with open(os.path.join(root, name), 'r+') as f:
            print (str(index))
            index += 1
            data = f.read()
            # lower case
            data = data.lower()
            # tokenization
            tokens = word_tokenize(data)
            # Remove punctuation tokens
            tokens = [word for word in tokens if word.isalpha()]
            # remove stop words
            stop_words = set(stopwords.words('english'))
            filtered_sentence = [w for w in tokens if not w in stop_words]
            lemmatizered_sentence = []
            for word in filtered_sentence:
                lemmatizered_sentence.append(lemmatizer.lemmatize(word))
            data = ' '.join(lemmatizered_sentence)
            f.seek(0)
            f.write(data)
            f.truncate()
            f.close()


# from sklearn.datasets import load_files
# Loading the data sets from disk
# data_train = load_files(os.path.join(path_to_corpus_folder,train_folder))
# data_test = load_files(os.path.join(path_to_corpus_folder,test_folder))