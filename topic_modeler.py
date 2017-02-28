import csv
import json
import nltk
import os
import pdb
import re
import shutil
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib import colors as mcolors
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from skimage import color
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, \
                                            HashingVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline


class StemTokenizer(object):
    BLACKLIST = ['http', 'https', 'www', 'jpg', 'png', 'com', 'disquscdn',
                 'uploads', 'images', 'blockquote', '2017', '02', 'youtu', 'im']

    def __init__(self, stem=False):
        self.stem = stem
        self.stemmer = SnowballStemmer('english')
        self.tokenizer = RegexpTokenizer(r'\w+')

    def __call__(self, doc):
        stop = stopwords.words('english') + self.BLACKLIST
        out = []
        for t in self.tokenizer.tokenize(doc):
            # exclude single-letter tokens
            if t not in stop and len(t) >= 2:
                # optional: do stemming
                if self.stem:
                    t = self.stemmer.stem(t)
                out.append(t)
        return out


class TopicModeler(object):
    TFIDF = 'tfidf'
    TF = 'tf'
    HASH = 'hash'
    HASH_IDF = 'hash-idf'
    NMF = 'nmf'  # Non-Negative Matrix Factorization
    LDA = 'lda'  # Latent Dirichlet Allocation

    def __init__(self, data, vector_type=TFIDF, model_type=NMF, n_features=1000,
                 n_topics=20):
        """
        Holds state for text processing and topic modeling.
        Vector type choices: 'tfidf', 'tf', 'hash'
        Model type choices: 'lda', 'nmf'
        """
        self.data = data
        self.vector_type = vector_type
        self.model_type = model_type
        self.n_features = n_features
        self.n_topics = n_topics

        self.build_docs()

    def build_docs(self):
        """
        Generate corpus of text from disjoint json files.
        """
        docs = defaultdict(dict)
        thread_docs = defaultdict(str)
        self.data.load()

        for forum, threads in self.data.get_forum_threads().iteritems():
            for tid in threads:
                # load data
                try:
                    with open('data/threads/%s.json' % tid) as f:
                        js = json.load(f)
                        full_text = '\n'.join([p['text'] for p in js])
                        docs[forum][tid] = full_text
                        thread_docs[tid] = full_text
                except IOError:
                    print 'data for thread', tid, 'is no good'
                    del self.data.thread_posts[tid]
                    save_json(self.data.thread_posts, 'thread_posts')

        # we don't need any defaultdict fuckery later on
        self.docs = dict(docs)
        self.thread_docs = dict(thread_docs)

    def sample_docs(self, n_docs=500):
        """
        Sample threads from forums proportional to the total number of comments
        in each forum
        """
        docs = []
        activity = self.data.get_forum_activity()
        forums = self.docs.keys()
        # TODO: should we sample proportional to sqrt or just the activity?
        probs = np.array([np.sqrt(activity[f]) for f in forums])
        probs /= sum(probs)

        for i in range(n_docs):
            # choose a random forum, then choose a random document from that
            # forum (with replacement)
            f = np.random.choice(forums, p=probs)
            docs.append(np.random.choice(self.docs[f].values()))

        return docs

    def vectorize(self, vec_type=None, forums=None, threads=None):
        """
        Fit a vectorizer to a set of documents and transform them into vectors.
        Documents taken from a set of forums, or threads, or sampled from the
        whole corpus (default)
        """
        # self.docs is a dict of dicts of strings. We want a list of strings.
        if threads:
            docs = [self.thread_docs[t] for t in threads]
        elif forums:
            docs = []
            for forum in forums:
                docs.append('\n'.join(self.docs[forum].values()))
        else:
            docs = self.sample_docs()

        print 'vectorizing', len(docs), 'documents of total length', \
            sum([len(d) for d in docs])/1000, 'KB'

        vec_type = vec_type or self.vector_type

        # generate hashing vectors
        if vec_type == self.HASH_IDF:
            hasher = HashingVectorizer(n_features=self.n_features,
                                       tokenizer=StemTokenizer(),
                                       stop_words='english',
                                       non_negative=True, norm=None,
                                       binary=False)
            self.vectorizer = make_pipeline(hasher, TfidfTransformer())

        elif vec_type == self.HASH:
            self.vectorizer = HashingVectorizer(n_features=self.n_features,
                                                tokenizer=StemTokenizer(),
                                                stop_words='english',
                                                non_negative=False, norm='l2',
                                                binary=False)

        else:
            # generate term-frequency, inverse-document-frequency vectors
            if vec_type == self.TFIDF:
                Vectorizer = TfidfVectorizer
            # generate plain term-frequency vector
            elif vec_type == self.TF:
                Vectorizer = CountVectorizer

            self.vectorizer = Vectorizer(max_df=0.95, min_df=2,
                                         max_features=self.n_features,
                                         tokenizer=StemTokenizer(),
                                         stop_words='english')

        return self.vectorizer.fit_transform(docs)

    def fit_model(self, vectors, model_type=None):
        model_type = model_type or self.model_type

        if model_type == self.NMF:
            self.model = NMF(n_components=self.n_topics, random_state=1,
                             alpha=.1, l1_ratio=.5)
        elif model_type == self.LDA:
            self.model = LatentDirichletAllocation(n_topics=self.n_topics,
                                                   max_iter=5,
                                                   learning_method='online',
                                                   learning_offset=50.,
                                                   random_state=0)
        else:
            raise model_type

        print 'fitting model of type', model_type, 'to', vectors.shape[0], 'with', \
            self.n_topics, 'topics'

        res = self.model.fit_transform(vectors)
        self.top_topics = sum(res) / len(res)

        # utility function for mapping top word features to their actual text
        best_feats = lambda fnames, feats: [fnames[i] for i in feats.argsort()[:-6:-1]]

        print
        print 'Topics:'

        vec_names = self.vectorizer.get_feature_names()
        self.topics = []
        for group in self.model.components_:
            topic = ', '.join(best_feats(vec_names, group))
            self.topics.append(topic)
            print '%d.' % len(self.topics), topic

    def train(self):
        """
        Train a topic model on the entire text corpus.
        """
        print 'building vectors...'
        vectors = self.vectorize()

        print 'fitting model...'
        self.fit_model(vectors)

    def predict_topics_forums(self, forums, verbose=False):
        docs = []
        for forum in forums[:]:
            if forum not in self.docs:
                print 'forum', forum, 'has no documents!'
                forums.remove(forum)
                continue
            forum_doc = '\n'.join(d for d in self.docs[forum].values())
            docs.append(forum_doc)

        if not docs:
            return

        vec = self.vectorizer.transform(docs)   # needs to be a list!!
        res = self.model.transform(vec)

        if verbose:
            for i, r in enumerate(res):
                print 'Topics for forum "%s":' % forums[i]
                for j, idx in enumerate(r.argsort()[:-6:-1]):
                    print '%d. (%.3f)' % (j+1, r[idx]), self.topics[idx]

        if not verbose or len(res) > 1:
            total = np.zeros(res.shape[1])
            normal = 0
            for i, f in enumerate(forums):
                #score = np.sqrt(self.data.get_forum_activity()[f])
                score = self.data.get_forum_activity()[f]
                normal += score
                total += res[i] * score

            # generate normal score scaled by the number of posts in each forum
            total /= normal

            # compare against the baseline
            total /= self.top_topics
            print
            print 'Top topics for group %s:' % forums
            for i, idx in enumerate(total.argsort()[:-6:-1]):
                print '%d. (%.3f)' % (i+1, total[idx]), self.topics[idx]

        return pd.DataFrame(index=forums, columns=self.topics, data=res)

    def predict_topics_threads(self, threads):
        docs = []
        for thread in threads[:]:
            if thread not in self.thread_docs:
                print 'thread', thread, 'documents not found!'
                threads.remove(thread)
                continue
            docs.append(self.thread_docs[thread])

        if not docs:
            return

        vec = self.vectorizer.transform(docs)   # needs to be a list!!
        res = self.model.transform(vec)

        for i, r in enumerate(res):
            print 'Top topics for thread on "%s":' % \
                self.data.all_threads[threads[i]]
            for j, idx in enumerate(r.argsort()[:-6:-1]):
                print '%d. (%.3f)' % (j+1, r[idx]), self.topics[idx]

        return pd.DataFrame(index=threads, columns=self.topics, data=res)

