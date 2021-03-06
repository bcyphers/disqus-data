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
#import matplotlib.pyplot as plt

from collections import defaultdict
#from matplotlib import colors as mcolors
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

from query_data import get_user_forums
from load_text import StemTokenizer
from functools import reduce


# utility function for mapping top word features to their actual text
def topic_name(fnames, feats):
    total = sum(feats)
    return  ', '.join('(%.2f) %s' % (feats[i] / total, fnames[i])
                      for i in feats.argsort()[:-6:-1])


def print_topics(vectorizer, model):
    print()
    print('Topics:')

    word_names = vectorizer.get_feature_names()
    topics = []
    for group in model.components_:
        topic = topic_name(word_names, group)
        topics.append(topic)
        print('%d.' % len(topics), topic)
    return topics


class TopicModeler(object):
    TFIDF = 'tfidf'     # term frequency-inverse document frequency
    TF = 'tf'           # plain old term frequency
    HASH = 'hash'       # terms are hashed into a smaller space (e.g. 1000)
    HASH_IDF = 'hash-idf'
    NMF = 'nmf'         # Non-Negative Matrix Factorization
    LDA = 'lda'         # Latent Dirichlet Allocation

    def __init__(self, data, vector_type=TFIDF, model_type=NMF, n_features=1000,
                 n_topics=40):
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

        forum_threads = [i for i in list(self.data.get_forum_threads().items()) if
                         len(i[1]) and i[0] in self.data.forum_details and
                         self.data.forum_details[i[0]]['language'] == 'en']
        self.docs = {f: ts for f, ts in forum_threads}
        self.threads = reduce(lambda x, y: x + y, list(self.docs.values()))

    def load_thread(self, tid):
        # load comments for thread
        try:
            with open('data/threads/%s.json' % tid) as f:
                js = json.load(f)
                full_text = '\n'.join([p['text'] for p in js])
                return full_text
        except IOError:
            print('data for thread', tid, 'is no good')
            del self.data.thread_posts[tid]
            return ''

    def load_forum_thread(self, forum):
        return '\n'.join((self.load_thread(t) for t in self.docs[forum]))

    def sample_docs(self, n_docs=1000):
        """
        Sample threads from forums proportional to the total number of comments
        in each forum
        """
        docs = []
        activity = self.data.get_forum_activity()
        forums = list(self.docs.keys())
        # TODO: should we sample proportional to sqrt or just the activity?
        probs = np.array([np.sqrt(activity[f]) for f in forums])
        probs /= sum(probs)

        for i in range(n_docs):
            # choose a random forum, then choose a random document from that
            # forum (with replacement)
            f = np.random.choice(forums, p=probs)
            tid = np.random.choice(self.docs[f])
            docs.append(self.load_thread(tid))

        return docs

    def vectorize(self, vec_type=None, forums=None, threads=None,
                  sample_size=1000):
        """
        Fit a vectorizer to a set of documents and transform them into vectors.
        Documents taken from a set of forums, or threads, or sampled from the
        whole corpus (default)
        """
        # self.docs is a dict of dicts of strings. We want a list of strings.
        if threads:
            docs = [self.load_thread(t) for t in threads]
        elif forums:
            docs = [self.load_forum_thread(forum) for forum in forums]
        else:
            docs = self.sample_docs(sample_size)

        print('vectorizing', len(docs), 'documents of total size', \
            sum([len(d) for d in docs])/1000, 'KB')

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

        print('fitting model of type', model_type, 'to', vectors.shape[0], 'with', \
            self.n_topics, 'topics')

        res = self.model.fit_transform(vectors)
        self.baseline_topics = sum(res) / len(res)

        self.topics = print_topics(self.vectorizer, self.model)

    def train(self, sample_size=1000):
        """
        Train a topic modeler on the entire text corpus.
        """
        print('building vectors...')
        vectors = self.vectorize(sample_size=sample_size)

        print('fitting model...')
        self.fit_model(vectors)

    def predict_topics_forums(self, forums, verbose=False):
        docs = []
        for forum in forums[:]:
            if forum not in self.docs:
                print('forum', forum, 'has no documents!')
                forums.remove(forum)
                continue
            docs.append(self.load_forum_thread(forum))

        if not docs:
            return

        vec = self.vectorizer.transform(docs)   # needs to be a list!!
        res = self.model.transform(vec)

        if verbose:
            for i, r in enumerate(res):
                print('Topics for forum "%s":' % forums[i])
                for j, idx in enumerate(r.argsort()[:-6:-1]):
                    print('%d. (%.3f)' % (j+1, r[idx]), self.topics[idx])

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
            total /= self.baseline_topics
            print()
            print('Top topics for group %s:' % forums)
            for i, idx in enumerate(total.argsort()[:-6:-1]):
                print('%d. (%.3f)' % (i+1, total[idx]), self.topics[idx])

        return pd.DataFrame(index=forums, columns=self.topics, data=res)

    def predict_topics_threads(self, threads):
        docs = []
        for thread in threads[:]:
            if thread not in self.threads:
                print('thread', thread, 'documents not found!')
                threads.remove(thread)
                continue
            docs.append(self.load_thread(thread))

        if not docs:
            return

        vec = self.vectorizer.transform(docs)   # needs to be a list!!
        res = self.model.transform(vec)

        for i, r in enumerate(res):
            print('Top topics for thread on "%s":' % \
                self.data.all_threads[threads[i]])
            for j, idx in enumerate(r.argsort()[:-6:-1]):
                print('%d. (%.3f)' % (j+1, r[idx]), self.topics[idx])

        return pd.DataFrame(index=threads, columns=self.topics, data=res)


def model_forums_as_topics(year=2017, cutoff=5, forum=None, n_topics=20):
    print('querying')
    user_docs = get_user_forums(year)

    print('sampling')
    # only include users who have posted in enough different forums, and exclude
    # anonymous users (uid == -1)
    sample = {u: d for u, d in user_docs.items()
              if len(set(d)) >= cutoff and u != -1}
    # number of posts each user made
    user_count = {}

    if forum is not None:
        for u, d in list(sample.items()):
            if forum in d:
                sample[u] = [i for i in d if i != forum]
                user_count[u] = len([i for i in d if i == forum])
            else:
                del sample[u]
        if len(sample) == 0:
            print("No posters in forum '%s' found!")
            return
    else:
        user_count = {u: len(d) for u, d in sample.items()}

    users, docs = list(zip(*list(sample.items())))
    print(len(users), 'users with', sum(user_count.values()), 'posts')
    strings = [' '.join(doc) for doc in docs]

    print('vectorizing')
    vectorizer = TfidfVectorizer(min_df=5, max_features=1000,
                                 preprocessor=lambda s: s,
                                 tokenizer=lambda s: s.split())
    vectors = vectorizer.fit_transform(strings)

    print('topic modeling')
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=10,
                                    learning_method='online',
                                    learning_offset=50., random_state=0)
    user_topics = {users[i]: vec for i, vec in
                   enumerate(lda.fit_transform(vectors))}
    topic_names = print_topics(vectorizer, lda)

    top_topics = {t: 0 for t in range(n_topics)}
    sum_topics = {t: 0 for t in range(n_topics)}
    for u, vec in user_topics.items():
        top_topic = np.argsort(vec)[-1]
        top_topics[top_topic] += user_count[u]
        for t, val in enumerate(vec):
            sum_topics[t] += val * user_count[u]

    print()
    print('Most common top topics:')
    for t, count in sorted(list(top_topics.items()), key=lambda i: -i[1]):
        print('%d: %s' % (count, topic_names[t]))

    print()
    print('Top topics overall:')
    for t, count in sorted(list(sum_topics.items()), key=lambda i: -i[1]):
        print('%.2f: %s' % (count, topic_names[t]))

    return strings, vectorizer, vectors, lda, users, user_topics
