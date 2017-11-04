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
from gensim.models.phrases import Phraser, Phrases
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
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select, MetaData
from sqlalchemy.sql import and_, or_, not_
from scipy.spatial.distance import cosine
from scipy.spatial import procrustes
from scipy.stats.stats import pearsonr
from scipy.optimize import curve_fit

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

from orm import *
from word2vec_align import smart_align_gensim, procrustes_align, align_vocab


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
        self.forum_biases = {}
        if forums is None:
            self.forums = []
            with open('./all_forums.txt') as f:
                for line in f:
                    elts = line.strip().split()
                    self.forums.append(elts[0])
                    if len(elts) > 1:
                        self.forum_biases[elts[0]] = int(elts[1])
        else:
            self.forums = forums

        self.start_time = start_time
        self.end_time = end_time
        self.limit = limit
        self.tokenize = StemTokenizer(False)
        self.stopwords = stopwords.words('english')
        self.partisanships = None
        self.similarities = None

    @property
    def words(self):
        self._words = self.models.values()[0].wv.index2word
        return self._words

    def load_data(self, forum, cache='./post_cache/3gram'):
        """
        Load all posts for forum between start_time and end_time.
        If present on the local file system, load and return that; otherwise
        pull data from the database.
        """
        posts = []
        fname = cache + '/%s.txt' % forum

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
                tstr = p + '\n'
                f.write(tstr.encode('utf8'))

        return posts

    def train_phraser(self, threshold=10.):
        """
        For the given forum, find a set of phrases (n-grams of words) that
        appear often enough together to be considered their own tokens.
        (https://radimrehurek.com/gensim/models/phrases.html#module-gensim.models.phrases)
        """
        bigrams = Phrases(threshold=threshold)
        trigrams = Phrases(threshold=threshold)
        for forum in self.forums:
            fname = './post_cache_1gram/%s.txt' % forum

            print 'loading data...'
            docs = self.load_data(forum)
            if len(docs) > 1e5:
                docs = [docs[i] for i in np.random.choice(len(docs),
                                                          size=int(1e5))]

            posts = [d.split() for d in docs]

            print 'updating bigrams...'
            bigrams.add_vocab(posts)

            print 'converting to bigrams...'
            posts = bigrams[posts]

            print 'updating trigrams...'
            trigrams.add_vocab(posts)

        return bigrams, trigrams

    def do_phrasing(self, forum, p2, p3):
        inf = './post_cache/1gram/%s.txt' % forum
        outf = './post_cache/3gram/%s.txt' % forum
        with open(inf) as infile:
            with open(outf, 'w') as outfile:
                for i, l in enumerate(infile):
                    p = p3[p2[l.decode('utf8').strip().split()]]
                    outfile.write((' '.join(p) + '\n').encode('utf8'))

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
            fname = './model_cache/3gram/%s.bin' % forum
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

    def align_embeddings(self):
        """
        After all models have been trained, align their vector spaces with
        Procrustes superimposition (https://en.wikipedia.org/wiki/Procrustes_analysis)
        so that we can compare embeddings across models.
        """
        print 'filtering vocabularies...'
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

        # make sure vocabs are lined up
        for k, m in items[-2::-1]:
            align_vocab(ref_model.wv.index2word, m.wv)
        return

        print 'finding average alignment...'

        for i in range(20):
            print i
            # find the "mean" shape of all the embeddings
            sums = np.zeros(ref_model.wv.syn0norm.shape)
            for m in self.models.values():
                sums += m.wv.syn0norm
            mean_shape = normalize(sums)

            for k, m in self.models.items():
                # align each model with the mean vector space
                m1, m2, score = procrustes(mean_shape, m.wv.syn0norm)
                m.wv.syn0norm = m.wv.syn0 = m2
                print k, score

    def score_partisanship_corr(self, word):
        forums, biases = zip(*self.forum_biases.items())
        series = []
        for f in forums:
            ix = self.models[f].wv.index2word.index(word)
            series.append(self.similarities[f][ix, :])
        series = np.array(series)
        err = sum(-np.log(pearsonr(series[:, i], biases)[1]) for i in
                  range(series.shape[1]))
        return err / series.shape[1]

    def compute_partisanship_corr(self):
        if self.partisanships is not None:
            return

        words = [w for w in self.words if w not in self.stopwords]
        self.partisanships = {}

        for i, w in enumerate(words):
            self.partisanships[w] = self.score_partisanship_corr(w)
            if (i+1) % 100 == 0:
                print "finished with %d words" % (i+1)

    def compute_partisanship_counts(self):
        if self.partisanships is not None:
            return

        forums, biases = zip(*self.forum_biases.items())
        counts = np.zeros((len(self.words), len(forums)))

        for i, f in enumerate(forums):
            vocab = self.models[f].wv.vocab
            total_count = sum(vocab[w].count for w in vocab)
            counts[:, i] = np.array([np.log(vocab[w].count / float(total_count))
                                     for w in self.words])

        self.partisanships = {w: pearsonr(counts[i, :], biases)[0] for i, w
                              in enumerate(self.words)}


    def compute_partisanship_cosine(self):
        """
        For each word, compute the average distance between its embeddings in any
        two models with the same bias vs. the average distance between any two
        models with different bias.
        """
        if self.partisanships is not None:
            return

        left = [f for f, b in self.forum_biases.items() if b <= 2]
        right = [f for f, b in self.forum_biases.items() if b >= 4]

        #models = {f: self.models[f].wv for f in left + right}
        models = self.map_to_same_space(left + right)

        print 'computing cross-class distances...'
        dist = np.zeros(len(self.words))
        count = 0
        for f1 in left:
            #m1 = self.models[f1].wv
            m1 = models[f1]
            for f2 in right:
                #m2 = self.models[f2].wv
                m2 = models[f2]
                for i, w in enumerate(self.words):
                    dist[i] += cosine(m1[w], m2[w])
                count += 1.
        dist /= count

        print 'computing intra-class distances...'
        same_dist = np.zeros(len(self.words))
        count = 0
        for forums in [right, left]:
            for i in range(len(forums)):
                #m1 = self.models[forums[i]].wv
                m1 = models[forums[i]]
                for j in range(i+1, len(forums)):
                    #m2 = self.models[forums[j]].wv
                    m2 = models[forums[j]]
                    for k, w in enumerate(self.words):
                        same_dist[k] += cosine(m1[w], m2[w])
                    count += 1.
        same_dist /= count
        diff = dist - same_dist

        self.partisanships = {w: diff[i] for i, w in enumerate(self.words)
                              if w not in self.stopwords}

    def get_similarities(self):
        print 'computing similarity matrices...'
        if self.similarities:
            print 'already done.'
        else:
            self.similarities = {}
            for f in self.forums:
                print f
                self.similarities[f] = cosine_similarity(self.models[f].wv.syn0)

        self.sim_index = {w: i for i, w in enumerate(self.models.values()[0].wv.index2word)}

    def get_word_proportions(self, words):
        fractions = defaultdict(float)
        print 'getting word weights...'
        for f in self.forum_biases:
            vocab = self.models[f].wv.vocab
            total_count = sum(vocab[w].count for w in words)
            for w in words:
                fractions[w] += vocab[w].count / float(total_count)

        for w in fractions:
            fractions[w] /= len(self.forum_biases)

        print 'done.'
        return fractions

    def get_word_counts(self, words):
        return {w: sum(self.models[f].wv.vocab[w].count for f in
                       self.forum_biases)
                for w in words}

    def map_to_same_space(self, forums, do_pca=True):
        models = {}
        self.get_similarities()

        mean_sims = np.zeros(self.similarities.values()[0].shape)
        for f in forums:
            s = self.similarities[f]
            mean_sims += s
        mean_sims /= len(forums)

        if not do_pca:
            for f in forums:
                models[f] = {w: self.similarities[f][i, :] for w, i in
                             self.sim_index.items()}
        else:
            print 'fitting PCA...'
            pca = PCA(n_components=100)
            pca.fit(mean_sims)

            print 'PCA transforming...'
            for f in forums:
                print f
                vectors = pca.transform(self.similarities[f])
                models[f] = {w: vectors[i, :] for w, i in self.sim_index.items()}

        return models

    def plot_partisanship(self):
        """
        Compute (if necessary) and plot the partisanship of each word in the
        combined models against its representation in the dataset.
        """
        combinations = []
        forums = self.forum_biases.keys()
        for i in range(len(forums)):
            for j in range(i+1, len(forums)):
                combinations.append((forums[i], forums[j]))

        self.compute_partisanship_counts()
        weights = self.get_word_proportions(self.words)
        words = sorted(self.words, key=lambda w: weights[w])

        # x is log(word counts), y is word "partisanship"
        x, y = np.array(zip(*[(np.log(weights[w]),
                               self.partisanships[w]) for w in words]))
        layout = go.Layout(
            title='Vector variation vs. word frequency',
            hovermode='closest',
            xaxis=dict(title='frequency', ticklen=5, gridwidth=2),
            yaxis=dict(title='partisanship', ticklen=5, gridwidth=2),
            showlegend=False)

        # graph x on a log-log scale (this is most aesthetically pleasing)
        trace = go.Scatter(x=x, y=y, mode='markers', name='words',
                           text=words)

        #def func(x, a, b, c):
            #return a * np.log(b * x) + c

        def lin_func(x, a, b):
            return a * x + b

        popt, pcov = curve_fit(lin_func, x, y)
        domain = np.linspace(min(x), max(x), 1000)
        fit_trace = go.Scatter(x=domain, y=lin_func(domain, *popt), mode='lines')

        py.plot(go.Figure(data=[trace, fit_trace], layout=layout),
                filename='average-model-distances')

    def find_partisan_correlations(self, word, do_shuf=False):
        dists = np.zeros((len(self.forum_biases), len(self.words)))
        biases = []
        forums = sorted(self.forum_biases.keys(), key=self.forum_biases.get)
        self.get_similarities()
        for i, (forum, bias) in enumerate(self.forum_biases.items()):
            biases.append(bias)
            model = self.models[forum]
            dists[i, :] = self.similarities[forum][self.words.index(word)]

        if do_shuf:
            np.random.shuffle(biases)

        print 'running regressions...'
        slopes = [tuple(np.polyfit(biases, dists[:, i], 1)) for i in range(dists.shape[1])]
        print 'computing correlations...'
        corrs = [tuple(pearsonr(dists[:, i], biases)) for i in range(dists.shape[1])]
        res = [(slopes[i], corrs[i], self.words[i]) for i in range(len(corrs))]
        print 'done.'

        return dists, res

    def plot_partisan_correlations(self, word, random=False):
        dists, res = self.find_partisan_correlations(word, do_shuf=random)
        fits, corrs, text = zip(*res)
        m = np.array(zip(*fits)[0])
        p = np.array(zip(*corrs)[0])
        text = np.array(text)
        weights = self.get_word_proportions(text)
        x = np.array([np.log(weights[w]) for w in text])

        left = np.array([i for i, v in enumerate(p) if v < 0])
        right = np.array([i for i, v in enumerate(p) if v > 0])
        left_p, right_p = p[left] ** 2, p[right] ** 2
        left_m, right_m = np.abs(m[left]), np.abs(m[right])
        left_t, right_t = text[left], text[right]

        print 'generating plot...'
        layout = go.Layout(
            title=word,
            hovermode='closest',
            xaxis=dict(title='correlation', ticklen=5, gridwidth=2),
            yaxis=dict(title='slope', ticklen=5, gridwidth=2),
            showlegend=False)

        left_trace = go.Scatter(x=left_p, y=left_m, mode='markers', text=left_t,
                                marker=dict(color='rgb(0,0,255)'))
        right_trace = go.Scatter(x=right_p, y=right_m, mode='markers', text=right_t,
                                 marker=dict(color='rgb(255,0,0)'))
        #trace = go.Scatter(x=x, y=np.array(p) ** 2, mode='markers', text=text)
        py.plot(go.Figure(data=[left_trace, right_trace], layout=layout), filename='m-v-corr')
        print 'done.'

    def plot_partisan_similarity(self, w1, w2):
        """ Plot the similarity score of two words against partisan bias """
        biases = []
        similarities = []
        forums = []
        for f, bias in self.forum_biases.items():
            forums.append(f)
            biases.append(bias)
            similarities.append(self.models[f].similarity(w1, w2))

        p, std = pearsonr(biases, similarities)
        title = '(%s, %s): p = %.3f' % (w1, w2, p)

        m, b = np.polyfit(biases, similarities, 1)
        fit = [m * x + b for x in range(1, 6)]

        print 'generating plot...'
        layout = go.Layout(
            title=title,
            hovermode='closest',
            xaxis=dict(title='partisan bias', ticklen=5, gridwidth=2),
            yaxis=dict(title='similarity', range=[-1, 1], ticklen=5, gridwidth=2),
            showlegend=False)

        trace = go.Scatter(x=biases, y=similarities, mode='markers', text=forums)
        fit_trace = go.Scatter(x=range(1, 6), y=fit, mode='lines')
        py.plot(go.Figure(data=[trace, fit_trace], layout=layout),
                filename='word-similarity')
        print 'done.'

    def plot_partisan_count(self, word):
        """ Plot the log probability of one word against partisan bias """
        forums = []
        biases = []
        counts = []
        for f, bias in self.forum_biases.items():
            forums.append(f)
            biases.append(bias)
            vocab = self.models[f].wv.vocab
            total_count = sum(vocab[w].count for w in self.words)
            counts.append(np.log(vocab[word].count / float(total_count)))

        p, std = pearsonr(biases, counts)
        title = '%s: p = %.3f' % (word, p)

        m, b = np.polyfit(biases, counts, 1)
        fit = [m * x + b for x in range(1, 6)]

        print 'generating plot...'
        layout = go.Layout(
            title=title,
            hovermode='closest',
            xaxis=dict(title='partisan bias', ticklen=5, gridwidth=2),
            yaxis=dict(title='proportion', ticklen=5, gridwidth=2),
            showlegend=False)

        trace = go.Scatter(x=biases, y=counts, mode='markers', text=forums)
        fit_trace = go.Scatter(x=range(1, 6), y=fit, mode='lines')
        py.plot(go.Figure(data=[trace, fit_trace], layout=layout),
                filename='word-proportion')
        print 'done.'

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

