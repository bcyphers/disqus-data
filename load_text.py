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

from bs4 import BeautifulSoup
from collections import defaultdict, OrderedDict
from gensim.models import Word2Vec
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
    BLACKLIST = ['http', 'https', 'www', 'jpg', 'png', 'com', 'disquscdn',
                 'net', 'uploads', 'images', 'blockquote', '2017', '01', '02',
                 'youtu', 'im']

    def __init__(self, stem=False):
        self.stem = stem
        self.stemmer = SnowballStemmer('english')
        self.tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)

    def __call__(self, doc):
        stop = stopwords.words('english') + self.BLACKLIST
        out = []
        sentences = nltk.tokenize.sent_tokenize(doc)
        for s in sentences:
            for t in self.tokenizer.tokenize(s):
                # exclude single-letter tokens
                if len(t) >= 2: # and t not in stop:
                    # optional: do stemming
                    if self.stem:
                        t = self.stemmer.stem(t)
                    out.append(t)
        return out


def clean_tokenize(text, stem=False):
    """ returns a list of tokens """
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    tokenize = StemTokenizer(stem)
    return tokenize(text)


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


def get_tokenized_posts(forum=None, author=None, adult=False):
    engine, session = get_mysql_session()

    print "querying for posts%s..." % ((' from forum ' + forum) if forum else '' +
                                       (' from user ' + author) if author else '')
    query = session.query(Post)
    if forum is not None:
        query = query.filter(Post.forum == forum)
    if author is not None:
        query = query.filter(Post.author == author)
    if adult:
        query = query.filter(Post.forum_pk == Forum.pk).\
                      filter(Forum.adult_content == 1)

    query = query.limit(5000000)
    df = pd.read_sql(query.statement, query.session.bind)

    print "creating post graph..."
    class P(object):
        def __init__(self, id, thread, parent, time):
            self.id = id
            self.thread = thread
            self.parent = parent
            self.time = time

    posts = {pid: P(pid, df.thread[pid], df.parent[pid], df.time[pid])
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
    return [clean_tokenize(df.raw_text[p.id]) for p in ordered_posts]


def get_forum_docs(forum):
    _, session = get_mysql_session()
    posts = session.query(Post.raw_text).filter(Post.forum == forum).all()
    fname = '%s-docs.txt' % forum
    with open(fname, 'w') as f:
        doc = u''
        for i, p in enumerate(posts):
            doc_str = clean_tokenize(p[0])
            doc += u'_*%d %s\n' % (i, doc_str)
            if int((100. * i) / len(posts)) > int((100. * (i-1)) / len(posts)):
                sys.stdout.write(' %d%% done\r' % (100. * i / len(posts)))
                sys.stdout.flush()

        f.write(doc.encode('utf8'))
        print 'docs written to', fname
