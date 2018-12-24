path_to_corpus_folder = 'ohsumed-first-20000-docs'
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

def process_corpus():
    lemmatizer = WordNetLemmatizer()
    ps = PorterStemmer()

    try:
        for root, dirs, files in os.walk(path_to_corpus_folder):
            for name in files:
                with open(os.path.join(root, name), 'r+') as f:
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
                    #Lemmatization
                    lemmatizered_sentence = []
                    for word in filtered_sentence:
                        lemmatizered_sentence.append(lemmatizer.lemmatize(word))
                    data = ' '.join(lemmatizered_sentence)

                    f.seek(0)
                    f.write(data)
                    f.truncate()
                    f.close()
    except:
        print(data)

# from sklearn.datasets import load_files
# Loading the data sets from disk
# data_train = load_files(os.path.join(path_to_corpus_folder,train_folder))
# data_test = load_files(os.path.join(path_to_corpus_folder,test_folder))

if __name__ == "__main__":
    #process_corpus() #should run only once
    categories = {}

    for root, dirs, files in os.walk(path_to_corpus_folder):
        index = 0
        ans = {}

        for name in files:
            with open(os.path.join(root, name), 'r+') as f:
                data = f.read()
                data = data.split(' ')

                for word in data:
                    if word in ans:
                        ans[word] = ans[word] + 1
                    else:
                        ans[word] = 1
            index += 1

        if 'test' in root and root[-3:] != 'est':
            categories.update({root[-3:]: index})
            from collections import defaultdict

            dct = defaultdict(list)

            for k, v in ans.items():
                dct[v].append(k)
            print('Top 10 words in ' + root[-3:])

            from prettytable import PrettyTable

            t = PrettyTable(['Word', '# of apps'])
            for k, v in sorted(dct.items(), reverse=True)[:10]:
                t.add_row([', '.join(v), k])

            #print the table
            print(t)

        if 'training' in root:
            break

    for key, value in categories.items():
        print('category: ' + key + ', # of docs: ' + str(value))