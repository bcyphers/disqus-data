import csv
import json
import nltk
import os
import pdb
import re
import shutil
import sys
import time
import warnings
import numpy as np
import pandas as pd

import url_parse

from bs4 import BeautifulSoup
from collections import defaultdict, OrderedDict
from gensim.models import Word2Vec
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
from skimage import color
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, \
                                            HashingVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline

from collect_data import *
from orm import *

# Ignore annoying warnings from BeautifulSoup
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


class StemTokenizer(object):
    URL_STOPS = ['http', 'https', 'www', 'jpg', 'png', 'com', 'disquscdn',
                 'net', 'uploads', 'images', 'blockquote', '2017', '01', '02',
                 'youtu', 'im', '__link__']

    def __init__(self, stem=False):
        self.stem = stem
        self.stemmer = SnowballStemmer('english')
        #self.tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        # contractions and hyphenations count as one word
        self.tokenizer = RegexpTokenizer('\w+(?:[/-\']\w+)*')
        self.language_counts = defaultdict(int)
        self.stopwords = stopwords.words('english') + self.URL_STOPS

    def __call__(self, doc):
        #try:
            #langs = [l.lang for l in detect_langs(doc)]
        #except LangDetectException as e:
            #print 'could not detect language for "%s".' %doc
            #return None

        #if 'en' not in langs:
            #return None

        out = []
        soup = BeautifulSoup(text, 'html.parser')
        doc = re.sub(url_parse.WEB_URL_REGEX, '__link__', soup.get_text())
        sentences = nltk.tokenize.sent_tokenize(doc)

        for s in sentences:
            for t in self.tokenizer.tokenize(s):
                # exclude single-letter tokens
                if len(t) >= 2: # and t not in stop:
                    # optional: do stemming
                    if self.stem:
                        t = self.stemmer.stem(t)
                    t = t.lower()
                    out.append(t)
        return out


tokenize = StemTokenizer(False)


class PostMeta(object):
    """ Small class to store post metadata for sorting, etc. """
    def __init__(self, id, thread, parent, time):
        self.id = id
        self.thread = thread
        self.parent = parent
        self.time = time


def order_thread_posts(posts):
    # order a thread's posts by time, preserving the DAG within the thread
    all_posts = sorted(posts.values(), key=lambda p: p.time)
    top_level_posts = []
    for p in all_posts:
        # children will be added in chronological order
        p.children = []
        if p.parent > 0 and p.parent in posts:
            posts[p.parent].children.append(p)
        else:
            top_level_posts.append(p)

    ordered_posts = []

    def append_recurse(post):
        ordered_posts.append(post)
        # now do all children in depth-first order
        for child in post.children:
            append_recurse(child)

    for p in top_level_posts:
        append_recurse(p)

    return ordered_posts


def get_tokenized_posts(forum=None, author=None, adult=False, start_time=None,
                        end_time=None, limit=5000000):
    Post = get_post_db(None)
    engine, session = get_mysql_session()

    print "querying for posts%s..." % ((' from forum ' + forum) if forum else '' +
                                       (' from user %s' % author) if author else '')
    query = session.query(Post)
    if forum is not None:
        query = query.filter(Post.forum == forum)
    if author is not None:
        if type(author) == list:
            query = query.filter(Post.author in authors)
        else:
            query = query.filter(Post.author == author)
    if adult:
        query = query.filter(Post.forum_pk == Forum.pk).\
                      filter(Forum.adult_content == 1)
    if start_time is not None:
        query = query.filter(Post.time >= start_time)
    if end_time is not None:
        query = query.filter(Post.time <= end_time)

    query = query.limit(limit)
    df = pd.read_sql(query.statement, query.session.bind)

    print "creating post graph..."

    posts = {pid: PostMeta(pid, df.thread[pid], df.parent[pid], df.time[pid])
             for pid in df.index}

    print len(posts), "found"
    print "ordering posts..."

    ordered_posts = []
    all_threads = OrderedDict()
    # need to order threads by time, and have a set of posts for each thread
    # (these can be ordered later)
    for p in posts.values():
        if p.thread not in all_threads:
            all_threads[p.thread] = {p.id: p}
        else:
            all_threads[p.thread][p.id] = p

    for t, tposts in all_threads.items():
        ordered_posts.extend(order_thread_posts(tposts))

    print "cleaning posts..."
    tokens = []
    for p in ordered_posts:
        t = tokenize(df.raw_text[p.id])
        if t is not None:
            tokens.append(t)
    return tokens


class VectorClassifier(object):
    """
    Use word2vec models trained on a variety of forum corpi to build a
    document classifier
    """
    def __init__(self, forums, start_time=None, end_time=None):
        # for each model, identify the words that differentiate it from the
        # others
        self.forums = forums
        self.start_time = start_time
        self.end_time = end_time
        self.forum_posts = {}
        self.train_models()

    def load_data(self, forum, limit=5000000):
        """ load all posts for forum between start_time and end_time """
        tokens = []
        fname = './post_cache/%s.txt' % forum
        self.forum_posts[forum] = []

        if os.path.isfile(fname):
            # load stuff if we can
            print 'loading posts for forum %s...' % forum
            with open(fname) as f:
                for line in f:
                    tokens.append(line.split())
        else:
            # otherwise query, clean, etc.
            print 'querying for posts for forum %s...' % forum
            Post = get_post_db(forum)
            engine, session = get_mysql_session()
            query = session.query(Post)
            if self.start_time is not None:
                query = query.filter(Post.time >= self.start_time)
            if self.end_time is not None:
                query = query.filter(Post.time <= self.end_time)

            query = query.limit(limit)
            df = pd.read_sql(query.statement, query.session.bind)

            print 'cleaning %d posts...' % len(df)
            for p in df.raw_text:
                t = tokenize(p)
                if t is not None:
                    tokens.append(t)

            print 'saving cleaned posts...'
            with open(fname, 'w') as f:
                for t in tokens:
                    tstr = ' '.join(t) + '\n'
                    f.write(tstr.encode('utf8'))

        self.forum_posts[forum] = tokens

    def train_models(self):
        self.models = {}
        for forum in self.forums:
            fname = './model_cache/%s.bin' % forum
            if os.path.isfile(fname):
                print 'loading model for forum %s...' % forum
                self.models[forum] = Word2Vec.load(fname)
                continue

            if forum not in self.forum_posts:
                self.load_data(forum)
            posts = self.forum_posts[forum]

            print 'training model for forum %s...' % forum
            # we need hs=1, negative=0 to do scoring (use hierarchical softmax,
            # no negative sampling)
            model = Word2Vec(posts, size=100, window=5, min_count=10,
                             workers=20, hs=1, negative=0)
            model.save(fname)
            self.models[forum] = model

    def train_model_vectors(self, num_words=1000, vector_size=1000):
        # what words are shared across all vocabs?
        models = self.models.values()
        overlap = set(models[0].wv.vocab.keys())
        for m in models:
            overlap &= set(m.wv.vocab.keys())

        # remove stopwords
        overlap = [w for w in overlap if w not in tokenize.stopwords]

        print 'overlap size: %d words' % len(overlap)

        # count the total number of words in each corpus
        model_counts = {f: sum([m.wv.vocab[w].count for w in m.wv.vocab])
                        for f, m in self.models.iteritems()}
        word_scores = {}
        for w in overlap:
            # average portion of each corpus that the word comprises
            word_scores[w] = sum([float(m.wv.vocab[w].count) / model_counts[f]
                                  for f, m in self.models.iteritems()])

        if num_words is not None:
            top_words = sorted(overlap, key=lambda w: -word_scores[w])[:num_words]
        else:
            top_words = overlap[:]

        probs = np.array([word_scores[w] for w in overlap])
        probs /= sum(probs)
        sample_vec = np.random.choice(overlap, vector_size, replace=False, p=probs)

        # dataframe for each model
        dfs = {}
        for f, m in self.models.iteritems():
            dfs[f] = pd.DataFrame(index=top_words, columns=sample_vec)
            for w in top_words:
                dfs[f].loc[w] = np.array([m.wv.similarity(w, w2) for w2 in
                                          sample_vec])

        print 'finished computing vectors for top %d words' % len(top_words)
        self.word_diffs = pd.Series(data=np.zeros(len(top_words)),
                                    index=top_words)

        for i in range(len(self.forums)):
            for j in range(i+1, len(self.forums)):
                f1 = self.forums[i]
                f2 = self.forums[j]
                for w in top_words:
                    self.word_diffs[w] += np.linalg.norm(dfs[f1].ix[w] -
                                                         dfs[f2].ix[w])
        self.dfs = dfs
        self.word_diffs = self.word_diffs.sort_values()


        # TODO: find word pairs whose similarity correlates with partisanship
        # find words whose overall shift in some direction correlates with partisanship
        # find embedding sets which facilitate partisan shift
        # e.g. which words' similarities to 'hillary' correlate with partisanship?

        # 1. build w2v models for ~20 forums with allsides ratings, over same
        # period of time


    def relevant(self, post):
        # is there at least one "indicator" word in the post?
        return len(set(post) & set(self.word_diffs[:100])) >= 1

    def score(self, posts):
        score = {}
        relevant = [p for p in posts if self.relevant(p)]
        for forum, m in self.models:
            score[forum] = m.score(relevant)

