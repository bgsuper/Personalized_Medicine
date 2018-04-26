from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import sklearn
import sklearn.naive_bayes, sklearn.ensemble
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import io, sys, os
import time, re

import requests, csv

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import gensim
import spacy
nlp = spacy.load('en')


class DataSet():

    def download_data(self, url, data_dir, name, **params):

        file_path = os.path.join(data_dir, name)

        if not os.path.isfile(file_path):
            print('Downloading File')
            data = requests.get(url)
            df = pd.read_csv(io.StringIO(data.text), **params)
            df.to_csv(file_path)
        else:
            print('Data exists')

    def show_documents(self, i=None):
        if i:
            return self.dataset.loc[[i]]
        else:
            return self.dataset.head()


    def remove_NAN(self, dataset_):
        dataset_=dataset_[pd.notnull(dataset_)]

    def keep_words(self, idx):
        self.text_vector = self.text_vector[:, idx]
        self.vocab = [self.vocab[i] for i in idx]


    def keep_top_words(self, M, Mprint=20):
        """Keep in the vocaluary the M words who appear most often."""
        freq = self.text_vector.sum(axis=0)
        freq = np.squeeze(np.asarray(freq))
        idx = np.argsort(freq)[::-1]
        idx = idx[:M]
        self.keep_words(idx)
        print('most frequent words')
        for i in range(Mprint):
            print('  {}: {} {} counts'.format(i, self.vocab[i], freq[idx][i]))
        return freq[idx]

    def lemmatize_(self, parsed_text):

        parsed_text = nlp(parsed_text)
        prof_token = [token.lemma_ for token in parsed_text
                      if token.is_oov
                      and not token.is_space
                      and not token.like_num
                      and not token.is_digit
                      and not token.suffix_.endswith('%')
                      and (token.is_ascii | any(u"\u03B1" <= c <= u"\u03C9" for c in token.text))
                      and not token.is_bracket
                      and not token.like_url and len(token) < 15]

        return ' '.join(prof_token)

    def lemmatize(self, dataset, col='Variation'):
        dataset[col+'_lemmatized'] = dataset.apply(lambda row: self.lemmatize_(row[col]), axis=1)

    def text_in_position(self, dataset, col):
        for index, row in dataset.iterrows():
            yield row[col]

    def vectorize(self, dataset_, **params):
        vectorizer = CountVectorizer(**params)
        self.text_vector = vectorizer.fit_transform(dataset_['text_prof_tokens'])
        self.vocab = vectorizer.get_feature_names()

    def vectorize_tfidf(self, dataset_, **params):
        vectorizer = TfidfVectorizer(**params)
        self.text_vector = vectorizer.fit_transform(dataset_['text_prof_tokens'])
        self.vocab = vectorizer.get_feature_names()
        [self.vocab.append(variation) for variation in dataset_['Variation_lemmatized'] if variation not in self.vocab]

    def remove_short_documents(self, dataset_, nwords):
        return dataset_[dataset_['text_length']> nwords]

    def normalize(self, norm='l1'):
        text_vector = self.text_vector.astype(np.float64)
        self.text_vector = sklearn.preprocessing.normalize(text_vector, axis=1, norm=norm)

    def data_info(self):
        N, M = self.text_vector.shape
        sparsity = self.text_vector.nnz / N / M * 100
        print('N = {} documents, M = {} words, sparsity={:.4f}%'.format(N, M, sparsity))

    def embed(self, filename=None, size=100):

        if filename:
            model = gensim.models.Word2Vec.load_word2vec_format(filename, binary=True)
            size = model.vector_size
        else:
            class Sentences():
                def __init__(self, documents):
                    self.documents=documents
                def __iter__(self):
                    for document in self.documents:
                        yield document.split()
            docs_list = [self.text.iloc[i,0] for i in range(self.text.shape[0])]
            model = gensim.models.Word2Vec(Sentences(docs_list), size, min_count=3)
            self.model=model
        self.embeddings = np.empty((len(self.vocab), size))

        keep = []
        not_found = 0
        for i, word in enumerate(self.vocab):
            try:
                self.embeddings[i, :] = model[word]
                keep.append(i)
            except KeyError:
                not_found += 1
        print('{} words not found in corpus'.format(not_found, i))

    def _extract_prof_text(self, row):
        parsed_text = nlp(row['Text'])

        prof_token = [token.lemma_ for token in parsed_text
                      if token.is_oov
                      and not token.is_space
                      and not token.like_num
                      and not token.is_digit
                      and not token.suffix_.endswith('%')
                      and (token.is_ascii | any(u"\u03B1" <= c <= u"\u03C9" for c in token.text))
                      and not token.is_bracket
                      and not token.like_url and len(token) < 15]

        prof_doc = ' '.join(prof_token)
        prof_doc = re.sub(r'[0]', ' ',prof_doc)
        prof_doc = ' '.join(re.findall(r'\b[a-zA-Z][a-zA-Z0-9]+\b', prof_doc))
        prof_doc = ' '.join([s for s in re.findall(r'\w+', prof_doc) if len(s)>2])

        return prof_doc

    def extract_prof_text(self):
        data_dir = './data'
        dataset_name = self.data_type

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
            dataset_name = dataset_name +'_clean_text.csv'
            dataset_clean_path = os.path.join(data_dir, dataset_name)
        else:
            dataset_name = dataset_name + '_clean_text.csv'
            dataset_clean_path = os.path.join(data_dir, dataset_name)

        if not os.path.isfile(dataset_clean_path):
            self.dataset_clean = self.dataset.copy()
            self.dataset_clean['text_prof_tokens'] = self.dataset_clean.apply(self._extract_prof_text, axis=1)
            self.dataset_clean = self.dataset_clean.dropna(how='any', axis=0)
            self.dataset_clean.to_csv(dataset_clean_path)
            self.text = pd.DataFrame.drop_duplicates(self.dataset_clean[['text_prof_tokens']])
        else:
            self.dataset_clean = pd.read_csv(dataset_clean_path)
            self.text = pd.DataFrame.drop_duplicates(self.dataset_clean[['text_prof_tokens']])

        self.dataset_clean.dropna(how='any', axis=0)
        self.lemmatize(self.dataset_clean)
        self.lemmatize(self.dataset_clean, col='Gene')
        self.dataset_clean = self.dataset_clean.drop(self.dataset_clean.columns[self.dataset_clean.columns.str.contains('Unnamed')], axis='columns')


    def text_length(self, dataframe):
        if 'text_length' not in dataframe.columns:
            dataframe['text_length'] = dataframe.apply(lambda row: len(row['text_prof_tokens'].split(' ')), axis=1)


class TrainDataSet(DataSet):

    def __init__(self, variants_dir, text_dir):
        train = pd.read_csv(variants_dir)
        trainx = pd.read_csv(text_dir, sep="\|\|", engine='python', header=None, skiprows=1,
                             names=["ID", "Text"])
        train = pd.merge(train, trainx, how='left', on='ID')

        self.dataset = train
        self.text = trainx
        self.data_type = 'train'

    def variation_class(self):
        df = self.dataset_clean[['Variation_lemmatized', 'Class']]
        return df


class TestDataSet(DataSet):

    def __init__(self, variants_dir, text_dir):

        test = pd.read_csv(variants_dir)
        testx = pd.read_csv(text_dir, sep="\|\|", engine='python', header=None, skiprows=1,
                            names=["ID", "Text"])
        test = pd.merge(test, testx, how='left', on='ID')
        self.dataset = test
        self.text = testx
        self.data_type = 'test'


def merge_text_data(df1, df2):
    col_name = df1.columns.values
    df_m = pd.DataFrame.drop_duplicates(df1.append(df2)).reset_index()
    df_m.drop(['index'])
    return df_m


def embedding(text_data):
    pass


def baseline(train_data, train_labels, test_data, test_labels, omit=[]):
    """Train various classifiers to get a baseline."""
    clf, train_accuracy, test_accuracy, train_f1, test_f1, exec_time = [], [], [], [], [], []
    clf.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=10))
    clf.append(sklearn.linear_model.LogisticRegression())
    clf.append(sklearn.naive_bayes.BernoulliNB(alpha=.01))
    clf.append(sklearn.ensemble.RandomForestClassifier())
    clf.append(sklearn.naive_bayes.MultinomialNB(alpha=.01))
    clf.append(sklearn.linear_model.RidgeClassifier())
    clf.append(sklearn.svm.LinearSVC())

    for i,c in enumerate(clf):
        if i not in omit:
            t_start = time.process_time()
            c.fit(train_data, train_labels)
            train_pred = c.predict(train_data)
            test_pred = c.predict(test_data)
            train_accuracy.append('{:5.2f}'.format(100*sklearn.metrics.accuracy_score(train_labels, train_pred)))
            test_accuracy.append('{:5.2f}'.format(100*sklearn.metrics.accuracy_score(test_labels, test_pred)))
            train_f1.append('{:5.2f}'.format(100*sklearn.metrics.f1_score(train_labels, train_pred, average='weighted')))
            test_f1.append('{:5.2f}'.format(100*sklearn.metrics.f1_score(test_labels, test_pred, average='weighted')))
            exec_time.append('{:5.2f}'.format(time.process_time() - t_start))

    print('Train accuracy:      {}'.format(' '.join(train_accuracy)))
    print('Test accuracy:       {}'.format(' '.join(test_accuracy)))
    print('Train F1 (weighted): {}'.format(' '.join(train_f1)))
    print('Test F1 (weighted):  {}'.format(' '.join(test_f1)))
    print('Execution time:      {}'.format(' '.join(exec_time)))


def grid_search(params, grid_params, train_data, train_labels, val_data,
        val_labels, test_data, test_labels, model):
    """Explore the hyper-parameter space with an exhaustive grid search."""
    params = params.copy()
    train_accuracy, test_accuracy, train_f1, test_f1 = [], [], [], []
    grid = sklearn.grid_search.ParameterGrid(grid_params)
    print('grid search: {} combinations to evaluate'.format(len(grid)))

    for grid_params in grid:
        params.update(grid_params)
        name = '{}'.format(grid)
        print('\n\n  {}  \n\n'.format(grid_params))
        m = model(params)
        m.fit(train_data, train_labels, val_data, val_labels)
        string, accuracy, f1, loss = m.evaluate(train_data, train_labels)
        train_accuracy.append('{:5.2f}'.format(accuracy)); train_f1.append('{:5.2f}'.format(f1))
        print('train {}'.format(string))
        string, accuracy, f1, loss = m.evaluate(test_data, test_labels)
        test_accuracy.append('{:5.2f}'.format(accuracy)); test_f1.append('{:5.2f}'.format(f1))
        print('test  {}'.format(string))

    print('\n\n')
    print('Train accuracy:      {}'.format(' '.join(train_accuracy)))
    print('Test accuracy:       {}'.format(' '.join(test_accuracy)))
    print('Train F1 (weighted): {}'.format(' '.join(train_f1)))
    print('Test F1 (weighted):  {}'.format(' '.join(test_f1)))

    for i,grid_params in enumerate(grid):
        print('{} --> {} {} {} {}'.format(grid_params, train_accuracy[i], test_accuracy[i], train_f1[i], test_f1[i]))


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()