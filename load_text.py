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
from datetime import datetime, timedelta
from gensim.models import Word2Vec
#from langdetect import detect, detect_langs
#from langdetect.lang_detect_exception import LangDetectException
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
from sqlalchemy import select, MetaData
from sqlalchemy.sql import and_, or_, not_

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

from orm import *
from word2vec_align import smart_align_gensim


TRUMP_START = datetime(2017, 1, 20, 17, 0, 0)

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
        self.tokenizer = RegexpTokenizer('\w+(?:[-/\']\w+)*')
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
        text = BeautifulSoup(doc, 'html.parser').get_text()

        # this line is necessary because links surrounded by lots of periods or
        # commas (......google.com,,,,,,,,,,,,,) break the url regex. Any
        # combination of two or more periods or commas is shortened.
        text = re.sub('\.[\.]+', '.', text)
        text = re.sub(',[,]+', ',', text)

        # replace unicode non-breaking spaces with normal spaces
        text = re.sub(u'\xa0', u' ', text)

        # replace all urls with __link__
        text = re.sub(url_parse.WEB_URL_REGEX, '__link__', text)

        sentences = nltk.tokenize.sent_tokenize(text)

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
                        end_time=None, limit=5000000, order=True):
    start_time = start_time or TRUMP_START
    Post = get_post_db(forum, start_time)
    engine, session = get_mysql_session()

    print "querying for posts%s..." % ((' from forum ' + forum) if forum else '' +
                                       (' from user %s' % author) if author else '')
    query = session.query(Post)
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

    if order:
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

        posts = [df.raw_text[p.id] for p in ordered_posts]

    else:
        posts = df.raw_text

    print "tokenizing posts..."
    tokenize = StemTokenizer(False)
    post_tokens = []
    for p in posts:
        tokens = tokenize(p)
        if tokens is not None:
            post_tokens.append(tokens)
    return post_tokens


class VectorClassifier(object):
    """
    Use word2vec models trained on a variety of forum corpi to build a
    document classifier. The goal is, for each Word2Vec model, to identify the
    words that differentiate it most from the other models.
    """
    def __init__(self, forums=None, start_time=None, end_time=None, limit=None):
        if forums is None:
            self.forums = []
            with open('./all_forums.txt') as f:
                for line in f:
                    self.forums.append(line.strip())
        else:
            self.forums = forums

        self.start_time = start_time
        self.end_time = end_time
        self.limit = limit
        self.tokenize = StemTokenizer(False)
        self.stopwords = stopwords.words('english')

    def load_data(self, forum):
        """
        Load all posts for forum between start_time and end_time.
        If present on the local file system, load and return that; otherwise
        pull data from the database.
        """
        posts = []
        fname = './post_cache/%s.txt' % forum

        if os.path.isfile(fname):
            # load stuff if we can
            print 'loading posts for forum %s...' % forum
            with open(fname) as f:
                for l in f:
                    posts.append(l.decode('utf8').strip())
            return posts

        # otherwise query, clean, etc.
        print 'querying for posts for forum %s...' % forum
        Post = get_post_db(forum=forum)
        engine, session = get_mysql_session()
        query = session.query(Post.id, Post.tokens)
        if self.start_time is not None:
            query = query.filter(Post.time >= self.start_time)
        if self.end_time is not None:
            query = query.filter(Post.time <= self.end_time)

        query = query.limit(self.limit)
        df = pd.read_sql(query.statement, query.session.bind)
        df.index = df.id

        posts = [t for t in df.tokens if t is not None]
        if not len(posts):
            print 'forum is not tokenized.'
            return None

        print 'saving cleaned posts...'
        with open(fname, 'w') as f:
            for p in posts:
                if p is not None:
                    tstr = t + '\n'
                    f.write(tstr.encode('utf8'))

        return posts

    def sub_relevant_ngrams(self, forum, min_frac=.2):
        """
        For the given forum, find a set of phrases (n-grams of words) that
        appear often enough together to be considered their own tokens. This is
        somewhat similar to the method used by gensim
        (https://radimrehurek.com/gensim/models/phrases.html#module-gensim.models.phrases)
        but we ignore stopwords and have a stricter threshold for phrases.
        """
        fname = './post_cache_tuples/%s.txt' % forum
        if os.path.isfile(fname):
            return None

        docs = self.load_data(forum)

        print 'training count vectorizer...'
        cv = CountVectorizer(ngram_range=(1,3), min_df=1e-5,
                             tokenizer=lambda l: l.split())
        if len(docs) > 1e6:
            doc_sample = [docs[i] for i in
                          np.random.choice(len(docs), size=int(1e6))]
        else:
            doc_sample = docs
        cv.fit(doc_sample)

        print 'transforming text sample...'
        countvec = np.zeros((1, len(cv.vocabulary_)))
        for i in range(0, len(doc_sample)/1000 + 1):
            combined = u' '.join(doc_sample[i*1000:(i+1)*1000])
            countvec += cv.transform([combined]).todense()

        print 'summing up counts...'
        counts = {w: countvec[0, i] for w, i in cv.vocabulary_.iteritems()}
        word_counts = defaultdict(int)
        for t, c in counts.iteritems():
            for w in t.split():
                word_counts[w] += c

        tuples = [w for w in counts if len(w.split()) > 1]
        filtered = []

        print 'filtering %d tuples...' % len(tuples)
        for t in tuples:
            words = t.split()
            if len([w for w in words if w not in self.stopwords]) <= 1:
                continue
            for w in t.split():
                if w not in self.stopwords and \
                        counts[t] < word_counts[w] * min_frac:
                    break
            else:
                filtered.append(t)

        print filtered

        print 'writing new tuples to file...'
        replacements = {t: t.replace(' ', '_') for t in filtered}
        rep = dict((re.escape(k), v) for k, v in
                   replacements.iteritems())
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m:
                           rep[re.escape(m.group(0))], text)

        with open(fname, 'w') as f:
            for d in docs:
                d = pattern.sub(lambda m: rep[re.escape(m.group(0))], d)
                f.write((d + '\n').encode('utf8'))

        return filtered

    def train_models(self, max_posts=int(5e6)):
        """
        Train Word2Vec models on each of the forums in self.forums.
        This will call load_data() on each forum to either load post data from
        the file system or query it from the database.
        If models already exist in ./model_cache/, those models will be loaded
        instead of re-trained.
        """
        self.models = {}
        for forum in self.forums:
            fname = './model_cache_ngram/%s.bin' % forum
            if os.path.isfile(fname):
                print 'loading model for forum %s...' % forum
                self.models[forum] = Word2Vec.load(fname)
                continue

            posts = self.load_data(forum)
            if posts is None:
                continue

            if len(posts) > max_posts:
                print 'Too many posts found (%d). Sampling %d posts...' % \
                    (len(posts), max_posts)
                posts = [posts[i] for i in
                         np.random.choice(len(posts), size=max_posts)]

            for i in range(len(posts)):
                posts[i] = posts[i].split()

            print 'training model for forum %s on %d posts' % (forum, len(posts))
            # we need hs=1, negative=0 to do scoring (use hierarchical softmax,
            # no negative sampling)
            model = Word2Vec(posts, size=100, window=5, min_count=10,
                             workers=32, hs=1, negative=0)
            model.save(fname)
            self.models[forum] = model

    def get_forum_biases(self):
        """
        Load allsides bias ratings from the database for each of the forums in
        self.forums. Store all ratings in self.forum_biases.
        """
        self.forum_biases = {}
        for f in self.models:
            engine, session = get_mysql_session()
            meta = MetaData(bind=engine, reflect=True)
            allsides_forums = meta.tables['allsides_forums']
            allsides = meta.tables['allsides']

            pk = session.query(Forum.pk).filter(Forum.id == f).first()
            if not pk:
                continue
            pk = pk[0]

            select_ = select([allsides, allsides_forums]).where(
                                  and_(allsides.c.id == allsides_forums.c.allsides_id,
                                       allsides_forums.c.forum_pk == pk))

            conn = engine.connect()
            res = conn.execute(select_).first()

            if res:
                self.forum_biases[f] = res[2]

    def align_embeddings(self):
        """
        After all models have been trained, align their vector spaces with
        Procrustes superimposition (https://en.wikipedia.org/wiki/Procrustes_analysis)
        so that we can compare embeddings across models.
        """
        # use the model with the biggest vocab size as the first reference
        items = sorted(self.models.items(),
                       key=lambda i: -len(i[1].wv.vocab.keys()))
        ref_model = items[0][1]
        for k, m in items[1:]:
            m.wv = smart_align_gensim(ref_model.wv, m.wv)

        # now go backwards and align everything with the last model
        ref_model = items[-1][1]
        for k, m in items[-2::-1]:
            m.wv = smart_align_gensim(ref_model.wv, m.wv)

    def partisanship(self, word):
        """
        Given a word, compute the average distance between its embeddings in any
        two models with the same bias vs. the average distance between any two
        models with different bias.
        """
        diff = 0
        for i in range(1, 6):
            forums = [f for f, b in self.forum_biases.items() if b == i]
            others = [f for f in self.forum_biases.keys() if f not in forums]
            count = 0
            dist = 0
            for f in forums:
                for o in others:
                    dist += cosine(self.models[f].wv[w], self.models[o].wv[w])
                    count += 1.
            dist /= count
            same_dist = 0
            for i in range(len(forums)):
                for j in range(i+1, len(forums)):
                    same_dist += cosine(self.models[forums[i]][w], self.models[forums[j]][w])
                    count += 1.
            same_dist /= count
            diff += dist - same_dist
        return diff / 5

    def plot_partisanship(self):
        combinations = []
        forums = self.forum_biases.keys()
        for i in range(len(forums)):
            for j in range(i+1, len(forums)):
                combinations.append((forums[i], forums[j]))

        vocab = self.models[forums[0]].wv.vocab
        counts = {w: sum(self.models[f].wv.vocab[w].count for f in forums)
                  for w in vocab}
        partisanships = {w: partisanship(w) for w in vocab}
        vocab = sorted(vocab, key=lambda v: counts[v])

        # x is word counts, y is word "partisanship"
        x, y = np.array(zip(*[(counts[w], partisanship[w]) for w in vocab]))

        layout = go.Layout(
            tile='Vector variation vs. word frequency',
            hovermode='closest',
            xaxis=dict(title='frequency', ticklen=5, gridwidth=2),
            yaxis=dict(title='cosine distance', ticklen=5, gridwidth=2),
            showlegend=False)
        # graph x on a log-log scale (this is most aesthetically pleasing)
        trace = go.Scatter(x=np.log(np.log(x)), y=y, mode='markers',
                            name='words', text=vocab)
        py.plot(go.Figure(data=[trace], layout=layout),
                filename='average-model-distances')

    def find_partisan_correlations(self, word):
        for i in range(1, 6):



    def remap_embeddings(self, num_words=1000, vector_size=1000):
        """
        Map word embeddings to a space defined by the distance between each word
        and a set of reference words inside its original embedding.
        """
        # what words are shared across all vocabs?
        models = self.models.values()
        overlap = set(models[0].wv.vocab.keys())
        for m in models:
            overlap &= set(m.wv.vocab.keys())

        # remove stopwords
        overlap = [w for w in overlap if w not in self.tokenize.stopwords]

        print 'overlap size: %d words' % len(overlap)

        # count the total number of words in each corpus
        model_counts = {f: sum([m.wv.vocab[w].count for w in m.wv.vocab])
                        for f, m in self.models.iteritems()}
        word_scores = {}
        for w in overlap:
            # average portion of each corpus that the word represents
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

    def relevant(self, post):
        # is there at least one "indicator" word in the post?
        return len(set(post) & set(self.word_diffs[:100])) >= 1

    def score(self, posts):
        score = {}
        relevant = [p for p in posts if self.relevant(p)]
        for forum, m in self.models:
            score[forum] = m.score(relevant)


# TODO: find word pairs whose similarity correlates with partisanship
# find words whose overall shift in some direction correlates with partisanship
# find embedding sets which facilitate partisan shift
# e.g. which words' similarities to 'hillary' correlate with partisanship?

# 1. build w2v models for ~20 forums with allsides ratings, over same
# period of time

# 2. Treat users as terms, threads as documents, perform LSA to find
# user types

# Unrelated: lots of political argument involves political script kiddies


class EmbeddingAligner(object):
    def __init__(self, forum):
        self.forum = forum
        self.models = {}
        self.windows = {}

    def build_models(self, start_time=TRUMP_START - timedelta(days=365),
                     end_time=TRUMP_START, delta=timedelta(days=30)):
        engine, session = get_mysql_session()
        Post = get_post_db(forum=self.forum)
        win_start = start_time
        win_end = win_start + delta
        idx = 0

        while win_start <= end_time - delta:
            self.windows[idx] = (win_start, win_end)
            fname = 'w2v_models/%s_%s_%s.bin' % (self.forum, win_start, delta)

            if os.path.isfile(fname):
                print 'loading model for forum %s, date %s...' % (self.forum,
                                                                  win_start)
                self.models[idx] = Word2Vec.load(fname)
            else:
                query = session.query(Post.id, Post.tokens)\
                    .filter(Post.time >= win_start)\
                    .filter(Post.time < win_end)
                print "querying for range (%s, %s)" % (win_start, win_end)

                posts = [i[1].split() for i in query.all()]

                print "building model"
                # we need hs=1, negative=0 to do scoring (use hierarchical softmax,
                # no negative sampling)
                model = Word2Vec(posts, size=100, window=5, min_count=10,
                                 workers=20, hs=1, negative=0)

                print "saving"
                model.save(fname, ignore=[])
                self.models[idx] = model

            win_start = win_end
            win_end += delta
            idx += 1

    def line_up_vectors(self):
        """ Use that funky transform to line up all the vector spaces """
        pass

    def plot_word_drift(self, word):
        """ plot a single word's drift through word-space """
        pass

    def plot_pair_drift(self, word1, word2):
        """ plot the change in the way two words relate to each other """
        for i, model in self.models.items():
            pass

