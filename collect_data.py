import csv
import json
import os
import pdb
import re
import shutil
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from disqusapi import DisqusAPI, APIError, FormattingError
from collections import defaultdict
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
        # file doesn't exist yet, so use default
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
        self.all_users = load_json('all_users', default={})
        self.done_with = set(load_json('done_with', default=[]))
        self.forum_threads = load_json('forum_threads', default={})
        self.thread_posts = load_json('thread_posts', default={})

        self.all_threads = {}
        for ts in self.forum_threads.values():
            self.all_threads.update({t: ts[t]['clean_title'] for t in ts})


    ###########################################################################
    ##  Users and forums  #####################################################
    ###########################################################################

    def pull_users(self, forum, min_users=1000):
        """
        Try to pull at least min_users of a forum's most active (public) users.
        TODO: this function is ugly
        """
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
                raise err
            except FormattingError as err:
                print err
                break

            users |= set(u['id'] for u in res if not
                         (u['isAnonymous'] or u['isPrivate']))

            # count how many of these people are private
            for u in res:
                if u['isAnonymous']:
                    continue
                self.all_users[u['id']] = u
                if u['isPrivate']:
                    num_private += 1

            print 'public:', len(users), 'private:', num_private

            # break if there are no more pages
            if i > 0 and not res.cursor['hasNext']:
                self.done_with.add(forum)
                break
            i += 1

        return list(users)

    def pull_forums_for_user(self, user_id):
        """
        Request a user's most active forums (up to 100)
        """
        try:
            res = self.api.request('users.listMostActiveForums', user=user_id, limit=100)
            return [r.get('id') for r in res]
        except APIError as err:
            print 'error on user id', user_id, err
            raise err

    def pull_all_user_forums(self, min_users):
        """
        Go through users in our forum-to-user mapping and, for each one, pull
        a list of the forums they're most active in.
        """
        forums = sorted(self.get_forum_activity().items(), key=lambda i: -i[1])

        # loop over forums in order of most active
        for forum in forums:
            # check how many of these users we already have
            total_users = self.forum_to_users[forum]:
            without_data = [u for u in total_users if u not in
                            self.user_to_forums]
            with_data = len(total_users) - len(without_data)

            # only get a few for each forum
            if with_data >= len(total_users) or with_data >= min_users:
                print 'forum', forum, 'has enough users:', with_data
                continue

            # for each of the most active users of this forum, find what forums
            # they're most active on
            for uid in without_data:
                print 'pulling most active forums for user', uid
                self.user_to_forums[uid] = self.pull_forums_for_user(uid)

            print 'saving user-forum data...'
            save_json(self.user_to_forums, 'user_to_forums')

    def pull_all_forum_users(self, min_users=1000):
        """
        Loop over forums and pull in active user lists for each one
        """
        # loop over forums in order of most active
        forums = sorted(self.get_forum_activity().items(), key=lambda i: -i[1])
        for forum in forums:
            print 'pulling most active users for forum', repr(forum)
            self.forum_to_users[forum] = self.pull_users(forum, min_users)

            print 'saving forum-user data...'
            save_json(self.forum_to_users, 'forum_to_users')
            save_json(list(self.done_with), 'done_with')
            save_json(self.all_users, 'all_users')

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
            ts = sorted(ts.items(), key=lambda i: -i[1]['postsInInterval'])[:25]
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
        forums = self.get_weights()
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
            weights = self.get_weights(dedup)
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

    def get_forum_threads(self):
        threads = {}
        for f, ts in self.forum_threads.items():
            threads[f] = [t for t in ts if t in self.thread_posts]
        return threads

    # TODO: this is dumb
    def get_weights(self, dedup=False):
        forums = defaultdict(float)

        for ftf in self.get_forum_edges(dedup=dedup).values():
            for f, v in ftf.items():
                forums[f] += v / float(sum(ftf.values()))

        return forums


if __name__ == '__main__':
    puller = DataPuller(sys.argv[1])

    activity = puller.get_forum_activity()
    threads = puller.get_forum_threads()

    for forum, n_posts in sorted(activity.items(), key=lambda i: -i[1])[:100]:
        color = 'green' if forum in puller.forum_to_users else 'red'
        n_users_tot = len(puller.forum_to_users[forum])
        n_users_dl = len([u for u in puller.forum_to_users[forum] if u in
                          puller.user_to_forums])
        n_threads = len(threads[forum])
        tup = (n_posts, n_users_dl, n_users_tot, n_threads)
        print colored(forum, color), '%d comments, %d/%d active users, %d threads' % tup

    puller.pull_all_forum_users()
    #puller.pull_all_threads()
    #puller.pull_all_posts()
