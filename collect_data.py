import csv
import json
import os
import pdb
import shutil
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from termcolor import colored
from disqusapi import DisqusAPI, APIError, FormattingError
from collections import defaultdict
from numpy.linalg import matrix_power

DEDUP = {
    'channel-theatlanticdiscussions': 'theatlantic',
    'theatlanticcities': 'theatlantic',
    'theatlanticwire': 'theatlantic',
    'in-focus': 'theatlantic',
    'bwbeta': 'bloomberg',
    'bloombergview': 'bloomberg',
    'pbsnewshourformrodeo': 'pbsnewshour',
}

def save_json(data, name):
    path = name + '.json'
    bakpath = name + '.bak.json'

    # create a backup
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

            # break if we get redundant data, not sure why this happens
            #if users | new_users == users:
                #self.done_with.add(forum)
                #break

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

    def pull_thread_posts(self, thread, num_posts=1000):
        pass

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
        forums = self.get_forum_weights()
        for f in forums.keys():
            if f in self.forum_threads:
                del forums[f]

        # loop indefinitely, gathering data
        while forums:
            forum = sorted(forums.items(), key=lambda i: -i[1])[0][0]
            self.pull_forum_threads(forum)
            del forums[forum]

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

    # quick and dumb way to weight the forums globally
    def get_forum_weights(self, dedup=False):
        forums = defaultdict(float)

        for ftf in self.get_forum_edges(dedup=dedup).values():
            for f, v in ftf.items():
                forums[f] += v / float(sum(ftf.values()))

        return forums

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
            weights = self.get_forum_weights(dedup)
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


def hierarchical_cluster(df):
    # TODO
    for i, arr in enumerate(np.corrcoef(df.values.T.astype(float))):
        a = arr[:]
        a[i] = 0
        print df.columns[i], colored(df.columns[a.argmax()], 'green'), max(a), \
            colored(df.columns[a.argmin()], 'red'), min(a)


def do_mcl(df, e, r):
    # perform MCL with expansion power parameter e and inflation parameter r
    # higher r -> more granular clusters
    # based on
    # https://www.cs.ucsb.edu/~xyan/classes/CS595D-2009winter/MCL_Presentation2.pdf
    mat = df.values.astype(float)

    converged = False
    while not converged:
        # expand
        last_mat = mat.copy()
        mat = matrix_power(mat, e)

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


def mcl_correlations(puller, e=2, r=3):
    df = puller.build_matrix()
    df = get_correlations(df)
    for c in df.columns:
        df[c] = df[c].map(lambda v: v**2)
        df[c] /= sum(df[c])

    mcl = do_mcl(df, e, r)
    print mcl


if __name__ == '__main__':
    puller = DataPuller(sys.argv[1])

    activity = puller.get_forum_activity()
    for i in sorted(activity.items(), key=lambda i: -i[1])[:100]:
        color = 'green' if i[0] in puller.forum_to_users else 'red'
        print colored('%s: %d' % i, color)

    #puller.pull_all_threads()
    puller.pull_all_forums(min_users=100)
