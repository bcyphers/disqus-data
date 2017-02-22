import csv
import json
import os
import pdb
import re
import shutil
import sys
import time
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from disqusapi import DisqusAPI, APIError, FormattingError
from collections import defaultdict
from gensim import corpora, models
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from numpy import linalg
from termcolor import colored


DEDUP = {
    'channel-theatlanticdiscussions': 'theatlantic',
    'theatlanticcities': 'theatlantic',
    'theatlanticwire': 'theatlantic',
    'in-focus': 'theatlantic',
    'bwbeta': 'bloomberg',
    'bloombergview': 'bloomberg',
    'pbsnewshourformsrodeo': 'pbsnewshour',
    'pj-instapundit': 'pj-media',
    'spectator-new-blogs': 'spectator-org',
    'theamericanspectator': 'spectator-org',
}


def save_json(data, name):
    path = name + '.json'
    bakpath = name + '.bak.json'

    # create a backup
    with open(path, 'a'): pass
    os.rename(path, bakpath)

    try:
        with open(path, 'w') as out:
            json.dump(data, out)
    except KeyboardInterrupt as e:
        print 'KeyboardInterrupt. Restoring backup file...'
        shutil.copyfile(bakpath, path)
        sys.exit(0)
    except Exception as e:
        print e


def load_json(name, default=None):
    path = name + '.json'
    bakpath = name + '.bak.json'
    try:
        with open(path) as f:
            data = json.load(f)
    except ValueError:
        # problem with the json
        with open(bakpath) as f:
            data = json.load(f)
    except:
        data = default

    return data


class DataPuller(object):
    def __init__(self, keyfile):
        with open(keyfile) as kf:
            key = kf.read().strip()

        self.api = DisqusAPI(key, None)
        self.load()

    def load(self):
        # load json files
        self.user_to_forums = load_json('user_to_forums', default={})
        self.forum_to_users = load_json('forum_to_users', default={})
        self.done_with = set(load_json('done_with', default=[]))
        self.forum_threads = load_json('forum_threads', default={})
        self.thread_posts = load_json('thread_posts', default={})

    ###########################################################################
    ##  Users and forums  #####################################################
    ###########################################################################

    def pull_users(self, forum, min_users=100):
        users = set(self.forum_to_users.get(forum, []))
        assert len(users) < min_users
        num_private = 0
        i = 0

        print 'trying to pull', min_users - len(users), 'more users for forum', forum
        while len(users) < min_users:
            cursor = '%d:0:0' % (i * 100)
            try:
                res = self.api.request('forums.listMostActiveUsers', forum=forum,
                                       cursor=cursor, limit=100)
            except APIError as err:
                print err
                break
            except FormattingError as err:
                print err
                break

            new_users = set(u['id'] for u in res if not (u['isAnonymous'] or
                                                         u['isPrivate']))

            # count how many of these people are private
            for u in res:
                if not u['isAnonymous'] and u['id'] not in users and u['isPrivate']:
                    num_private += 1

            # set operations woohoo
            users |= new_users

            # break if there are no more pages
            if i > 0 and not res.cursor['hasNext']:
                self.done_with.add(forum)
                break

            i += 1
            print 'public:', len(users), 'private:', num_private

        return list(users)

    def pull_forums_for_user(self, user_id):
        try:
            res = self.api.request('users.listMostActiveForums', user=user_id, limit=100)
            return [r.get('id') for r in res]
            return [r.get('id') for r in res]
        except APIError as err:
            print 'error on user id', user_id, err
            return None

    def pull_forum_users(self, forum, min_users=100):
        print 'pulling most active users for forum', repr(forum)
        self.forum_to_users[forum] = self.pull_users(forum, min_users=1000)

        print 'saving forum data...'
        save_json(self.forum_to_users, 'forum_to_users')
        save_json(list(self.done_with), 'done_with')

        # for each of the most active users of this forum, find what forums
        # they're most active on
        for uid in self.forum_to_users[forum]:
            if uid in self.user_to_forums:
                continue

            # only pull data for min_users user IDs
            num_users = len([u for u in self.forum_to_users[forum] if u in
                             self.user_to_forums])
            if num_users >= min_users:
                print 'forum', forum, 'has enough users:', num_users
                break

            print 'pulling most active forums for user', uid

            # try to get data for this guy
            uf = self.pull_forums_for_user(uid)
            if uf is None:
                # most likely hit api limit
                break

            # store list of forums this user's active in
            self.user_to_forums[uid] = uf

        print 'saving user data...'
        save_json(self.user_to_forums, 'user_to_forums')

    def pull_all_forums(self, min_users):
        # remove forums we've collected enough data on
        def cull_forums(forums):
            for f, users in self.forum_to_users.items():
                if len([u for u in users if u in self.user_to_forums]) >= \
                        min_users or f in self.done_with:
                    del forums[f]
            return forums

        forums = cull_forums(self.get_forum_activity())

        # loop indefinitely, gathering data
        while forums:
            forum = sorted(forums.items(), key=lambda i: -i[1])[0][0]
            self.pull_forum_users(forum, min_users)

            # update forum rankings with new data
            forums = cull_forums(self.get_forum_activity())

    ###########################################################################
    ##  Threads and posts  ####################################################
    ###########################################################################

    def pull_thread_posts(self, thread, total_posts=1000):
        assert thread not in self.thread_posts

        print 'pulling first', total_posts, 'posts for thread', thread

        # pull first post in thread
        res = self.api.request('threads.listPosts', thread=thread,
                               order='asc', limit=1)
        self.thread_posts[thread] = [res[0]['id']]
        has_next = res.cursor['hasNext']
        cursor = res.cursor['next']
        num_posts = 1
        all_data = []

        while has_next and num_posts < total_posts:
            try:
                res = self.api.request('threads.listPosts', thread=thread,
                                       limit=100, cursor=cursor)
            except APIError as err:
                print err
                return
            except FormattingError as err:
                print err
                return

            # have to go backwards here because we want them in chron order
            has_next = res.cursor['hasPrev']
            cursor = res.cursor['prev']

            # reverse the order and save
            posts = list(res)[::-1]
            for p in posts:
                if p['id'] not in self.thread_posts[thread]:
                    self.thread_posts[thread].append(p['id'])

            # count number of posts
            num_posts = len(self.thread_posts[thread])
            print 'retrieved', num_posts, 'posts'

            for p in posts:
                dic = {'id': p['id'],
                       'text': p['raw_message'],
                       'author': p['author'].get('id', -1),
                       'time': p['createdAt'],
                       'points': p['points']}
                all_data.append(dic)

        print 'saving thread data...'
        with open('threads/%s.json' % thread, 'w') as f:
            # save the thread in its own file
            json.dump(all_data, f)
        save_json(self.thread_posts, 'thread_posts')

    def pull_all_posts(self):
        threads = []
        for ts in self.forum_threads.values():
            # only first ten per forum for now
            ts = sorted(ts.items(), key=lambda i: -i[1]['postsInInterval'])[:10]
            threads.extend([t for i, t in ts if i not in self.thread_posts])

        # do longest threads first
        threads.sort(key=lambda t: -t['postsInInterval'])

        # loop indefinitely, gathering data
        for thread in threads:
            print 'pulling data for thread', repr(thread['clean_title']), \
                'from forum', thread['forum']
            self.pull_thread_posts(thread['id'])

    def pull_forum_threads(self, forum):
        print 'pulling most popular threads for forum', forum
        assert forum not in self.forum_threads

        try:
            res = self.api.request('threads.listPopular', forum=forum,
                                   interval='30d', limit=100)
        except APIError as err:
            print err
            return
        except FormattingError as err:
            print err
            return

        # count number of threads
        num_posts = sum([t['postsInInterval'] for t in res])
        self.forum_threads[forum] = {t['id']: t for t in res}
        print 'retrieved', len(res), 'threads with', num_posts, 'posts'

        print 'saving thread data...'
        save_json(self.forum_threads, 'forum_threads')

    def pull_all_threads(self):
        forums = get_weights(self)
        for f in forums.keys():
            if f in self.forum_threads:
                del forums[f]

        # loop indefinitely, gathering data
        while forums:
            forum = sorted(forums.items(), key=lambda i: -i[1])[0][0]
            self.pull_forum_threads(forum)
            del forums[forum]

    ###########################################################################
    ##  Utility functions and graph stuff  ####################################
    ###########################################################################

    def get_deduped_ftu(self):
        ftu = {}
        for forum, users in self.forum_to_users.items():
            if forum in DEDUP:
                users = set(users) | set(self.forum_to_users[DEDUP[forum]])
                ftu[DEDUP[forum]] = list(users)
            else:
                ftu[forum] = users

        return ftu

    def get_deduped_utf(self):
        utf = {}
        for user, forums in self.user_to_forums.items():
            deduped = set(f for f in forums if f not in DEDUP)
            deduped |= set(DEDUP[f] for f in forums if f in DEDUP)
            utf[user] = list(deduped)

        return utf

    def get_forum_edges(self, dedup=True):
        forum_edges = {}
        if dedup:
            forum_to_users = self.get_deduped_ftu()
            user_to_forums = self.get_deduped_utf()
        else:
            forum_to_users = self.forum_to_users
            user_to_forums = self.user_to_forums

        for forum, users in forum_to_users.items():
            # this will map forums to counts - the number of this forum's top
            # users who also frequent each other forum
            out_counts = defaultdict(int)

            # iterate over all the top users of this forum
            for uid in users:
                # update the counts for all forums this top user is active in
                for f in user_to_forums.get(uid, []):
                    out_counts[f] += 1

            forum_edges[forum] = out_counts

        return forum_edges

    # map forums to recent activity
    def get_forum_activity(self, dedup=False):
        counts = defaultdict(int)
        for f, threads in self.forum_threads.items():
            counts[f] = sum([t['postsInInterval'] for t in threads.values()])

        return counts

    # generate a transition matrix for this graph
    def build_matrix(self, dedup=True, N=None):
        if dedup:
            forum_to_users = self.get_deduped_ftu()
        else:
            forum_to_users = self.forum_to_users

        edges = self.get_forum_edges(dedup)
        forums = [k for k in edges.keys() if len(forum_to_users[k]) > 0]
        if N is not None:
            weights = get_weights(self)
            forums = sorted(forums, key=lambda f: -weights[f])[:N]

        df = pd.DataFrame(columns=forums, index=forums)

        # iterate over all forums that we have outgoing data for
        for f1 in forums:
            for f2 in forums:
                # number of top users of forum f1 for whom f2 is a top forum
                # think of it like the weight of the edge from f1 to f2
                users_of_f2 = edges[f1].get(f2, 0)
                df[f1][f2] = float(users_of_f2)

            # column normalize
            df[f1] /= sum(df[f1])

        return df


def pagerank(df, iters = 10):
    for c in df.columns:
        df[c][c] = 0
    A = df.values.T.astype(float)
    n = A.shape[1]
    w, v = linalg.eig(A)
    vec = abs(np.real(v[:n, 0]) / linalg.norm(v[:n, 0], 1))
    ranks = {df.columns[i]: vec[i] for i in range(len(vec))}
    return sorted(ranks.items(), key=lambda i: i[1])


def get_weights(puller, dedup=False):
    forums = defaultdict(float)

    for ftf in puller.get_forum_edges(dedup=dedup).values():
        for f, v in ftf.items():
            forums[f] += v / float(sum(ftf.values()))

    return forums


def get_correlations(df):
    # Generate a correlation matrix for edge graph
    out = pd.DataFrame(columns=df.columns, index=df.index)
    for i, arr in enumerate(np.corrcoef(df.values.T.astype(float))):
        for j in range(len(arr)):
            out[df.columns[j]][df.columns[i]] = arr[j]
    return out


def print_correlations(df):
    for i, arr in enumerate(np.corrcoef(df.values.T.astype(float))):
        a = arr[:]
        a[i] = 0
        print df.columns[i], colored(df.columns[a.argmax()], 'green'), max(a), \
            colored(df.columns[a.argmin()], 'red'), min(a)


def get_correlation_graph(puller):
    df = puller.build_matrix()
    df = get_correlations(df)
    for c in df.columns:
        df[c] = df[c].map(lambda v: v**2)
        df[c] /= sum(df[c])

    return df


def hierarchical_cluster(df):
    pass # TODO


def do_mcl(df, e, r, subset=None):
    # perform the Markov Cluster Algorithm (mcl) with expansion power parameter
    # e and inflation parameter r
    # higher r -> more granular clusters
    # based on
    # https://www.cs.ucsb.edu/~xyan/classes/CS595D-2009winter/MCL_Presentation2.pdf
    if subset:
        df = df[subset].ix[subset]
        for c in df.columns:
            df[c] /= sum(df[c])

    mat = df.values.astype(float)

    converged = False
    while not converged:
        # expand
        last_mat = mat.copy()
        mat = linalg.matrix_power(mat, e)

        # inflate
        for i in range(mat.shape[0]):
            mat[i] **= r
            mat[i] /= sum(mat[i])

            for j in range(mat.shape[1]):
                if mat[i, j] < 1.0e-100:
                    mat[i, j] = 0

        # converge?
        if np.all(last_mat == mat):
            converged = True

    clusters = {}
    for i in range(mat.shape[1]):
        if sum(mat[:,i]) > 0:
            cluster = [j for j in range(mat.shape[0]) if mat[j,i] > 0]
            clusters[df.columns[i]] = sorted(map(lambda k: df.columns[k],
                                                 cluster))

    return clusters


def get_documents(puller, forum):
    threads = {f: [t for t in ts if t in puller.thread_posts] for f, ts in
               puller.forum_threads.items()}

    print 'generated dictionary'

    tokenizer = RegexpTokenizer(r'\w+')
    sw = stopwords.words('english')
    stemmer = PorterStemmer()

    texts = []
    for tid in threads[forum]:
        # load data
        with open('threads/%s.json' % tid) as f:
            js = json.load(f)
            text ='\n'.join([p['text'] for p in js])

        print 'loaded data for thread', tid

        # tokenize, stop words, stemming
        tokens = tokenizer.tokenize(text)
        clean_tokens = []
        for t in tokens:
            if t.lower() not in sw:
                clean_tokens.append(stemmer.stem(t))

        texts.append(clean_tokens)

        print 'cleaned data for thread', tid

    # TODO: do we need a whole separate library to do just this part?
    dic = corpora.Dictionary(texts)
    corpus = [dic.doc2bow(text) for text in texts]
    return dic, corpus

    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dic,
                                        passes=20)



if __name__ == '__main__':
    puller = DataPuller(sys.argv[1])

    activity = puller.get_forum_activity()
    for i in sorted(activity.items(), key=lambda i: -i[1])[:100]:
        color = 'green' if i[0] in puller.forum_to_users else 'red'
        print colored('%s: %d' % i, color)

    #puller.pull_all_forums(min_users=100)
    #puller.pull_all_threads()
    puller.pull_all_posts()
