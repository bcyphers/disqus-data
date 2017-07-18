import argparse
import awis
import csv
import dateutil.parser
import datetime
import json
import os
import pdb
import re
import shutil
import sys
import time

from dateutil.relativedelta import relativedelta
from disqusapi import DisqusAPI, APIError, FormattingError
from orm import get_post_db, Forum, Base, get_mysql_session
from query_doer import get_forum_counts
from collections import defaultdict
from numpy import linalg
from termcolor import colored
from sqlalchemy.sql import func


DEDUP = {
    'channel-theatlanticdiscussions': 'theatlantic',
    'theatlanticcities': 'theatlantic',
    'theatlanticwire': 'theatlantic',
    'in-focus': 'theatlantic',
    'bwbeta': 'bloomberg',
    'bloombergview': 'bloomberg',
    'pbsnewshourformsrodeo': 'pbsnewshour',
    'pj-instapundit': 'pj-media',
    'spectator-new-blogs': 'spectator-new-www',
    'spectatorwww': 'spectator-new-www',
    'theamericanspectator': 'spectator-org',
    'spectatororg': 'spectator-org',
    'channel-theavclubafterdark': 'avclub',
    'mtonews': 'mtocom',
}

# TODO: this is ugly
DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/data/'
print DATA_PATH

# the first instant of President Trump's tenure
TRUMP_START = datetime.datetime(2017, 1, 20, 17, 0, 0)

# arguments for the script: more to come
ap = argparse.ArgumentParser()
ap.add_argument('keyfile', type=str, nargs='+', help='path to api key file')
ap.add_argument('--data-path', type=str, default=DATA_PATH,
                help='path to data directory')
ap.add_argument('--remote', action='store_true',
                help='connect to remote database')
ap.add_argument('--forum', type=str, default=DATA_PATH,
                help='forum from which to collect data')
ap.add_argument('--start-time', type=str, default='2007-01-01T00:00:00',
                help='starting timestamp')
ap.add_argument('--end-time', type=str, default='2017-01-20T16:59:59',
                help='ending timestamp')


###############################################################################
# Stateless utility functions for manipulating data
###############################################################################

def save_json(data, name):
    path = DATA_PATH + name + '.json'
    bakpath = DATA_PATH + name + '.bak.json'

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
    path = DATA_PATH + name + '.json'
    bakpath = DATA_PATH + name + '.bak.json'
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


def add_month(dt):
    """
    add exactly one month to the datetime and return a new one
    why there isn't a built in solution is beyond me
    """
    return (dt.replace(day=1) + datetime.timedelta(days=31)).replace(day=dt.day)


class DataPuller(object):
    """
    Holds state for api and database connections and keeps data in memory

    This is big and ugly and inefficient -- all the data should not be held in
    memory like this unless absolutely necessary. I'm gradually moving form a
    JSON-based to a MySQL-based data solution.

    In genera, the _foo_bar attributes are variables that point to data sets
    (usually dictionaries) in memory. They should only be accessed as properties
    like instance.foo_bar, which will return the data in _foo_bar if it's
    already in memory or load it from disk if necessary. This prevents having a
    long, expensive memory load during initialization.
    """
    def __init__(self):
        self._user_to_forums = None
        self._forum_to_users = None
        self._all_users = None
        self._done_with = None
        self._all_forum_threads = None
        self._active_forum_threads = None
        self._thread_posts = None
        self._forum_details = None

        self.forum_counts = None
        _, self.session = get_mysql_session(remote=args.remote)

    def __del__(self):
        self.session.close()

    def load_key(self, keyfile):
        """ Initialize a DisqusAPI handle """
        with open(keyfile) as kf:
            key = kf.read().strip()

        self.api = DisqusAPI(key, None)

    ###########################################################################
    # All these properties are this way so that big json files are only loaded
    # on demand
    ###########################################################################

    @property
    def user_to_forums(self):
        if self._user_to_forums is None:
            print 'loading user-forums'
            self._user_to_forums = load_json('user_to_forums', default={})
        return self._user_to_forums

    @property
    def forum_to_users(self):
        if self._forum_to_users is None:
            print 'loading forum-users'
            self._forum_to_users = load_json('forum_to_users', default={})
        return self._forum_to_users

    @property
    def all_users(self):
        if self._all_users is None:
            print 'loading user data'
            self._all_users = load_json('all_users', default={})
        return self._all_users

    @property
    def done_with(self):
        if self._done_with is None:
            print 'loading done with'
            self._done_with = set(load_json('done_with', default=[]))
        return self._done_with

    @property
    def all_forum_threads(self):
        if self._all_forum_threads is None:
            print 'loading all forum threads'
            self._all_forum_threads = load_json('all_forum_threads', default={})
        return self._all_forum_threads

    @property
    def active_forum_threads(self):
        if self._active_forum_threads is None:
            print 'loading top forum threads'
            self._active_forum_threads = load_json('active_forum_threads', default={})
        return self._active_forum_threads

    @property
    def thread_posts(self):
        if self._thread_posts is None:
            print 'loading top thread posts'
            self._thread_posts = load_json('thread_posts', default={})
        return self._thread_posts

    @property
    def forum_details(self):
        if self._forum_details is None:
            self._forum_details = load_json('forum_details', default={})
        return self._forum_details

    @property
    def all_threads(self):
        """
        build a dictionary mapping thread id to thread details for all
        threads we have downloaded
        """
        if self._all_threads is None:
            self._all_threads = {}
            for ts in self.active_forum_threads().values():
                self.all_threads.update({t: ts[t]['clean_title'] for t in ts})
        return self._all_threads

    def save_exit(self):
        """ Save all state and exit safely """
        print 'saving all data...'

        # save all json files
        if self._user_to_forums:
            save_json(self._user_to_forums, 'user_to_forums')
        if self._forum_to_users:
            save_json(self._forum_to_users, 'forum_to_users')
        if self._all_users:
            save_json(self._all_users, 'all_users')
        if self._done_with:
            save_json(list(self._done_with), 'done_with')
        if self._all_forum_threads:
            save_json(self._all_forum_threads, 'all_forum_threads')
        if self._active_forum_threads:
            save_json(self._active_forum_threads, 'active_forum_threads')
        if self._thread_posts:
            save_json(self._thread_posts, 'thread_posts')
        if self._forum_details:
            save_json(self._forum_details, 'forum_details')

        sys.exit(0)

    ###########################################################################
    ##  Users and forums  #####################################################
    ###########################################################################

    def pull_users(self, forum, min_users=1000):
        """ Try to pull at least min_users of a forum's most active users. """
        # TODO: this function is ugly
        users = set(self.forum_to_users.get(forum, []))
        assert len(users) < min_users
        num_private = 0
        i = 0
        errs = 0

        print 'trying to pull', min_users - len(users), \
            'more users for forum', forum

        while len(users) < min_users:
            cursor = '%d:0:0' % (i * 100)
            try:
                res = self.api.request('forums.listMostActiveUsers', forum=forum,
                                       cursor=cursor, limit=100)
            except APIError as err:
                raise err
            except FormattingError as err:
                print err
                errs += 1
                i += 1
                if errs > 2:
                    # give up on this shit
                    self.done_with.add(forum)
                    break
                continue

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

    def pull_all_user_forums(self, min_users):
        """
        Go through users in our forum-to-user mapping and, for each one, pull
        a list of the forums they're most active in.
        """
        activity = self.get_forum_activity()
        all_forums = [f for f in self.get_all_forums() if f in self.forum_to_users]
        forums = sorted(all_forums, key=lambda i: -activity.get(i, 0))

        # loop over forums in order of most active
        for forum in forums:
            # check how many of these users we already have
            total_users = self.forum_to_users[forum]
            without_data = [u for u in total_users if u not in
                            self.user_to_forums]
            with_data = len(total_users) - len(without_data)

            # only get a few for each forum
            if with_data >= len(total_users) or with_data >= min_users:
                print 'forum', forum, 'has enough users:', with_data
                continue

            to_go = min_users - with_data

            print
            print 'pulling data for', to_go, 'users from forum', forum, 'with',\
                activity.get(forum), 'posts in 30 days'

            # for each of the most active users of this forum, find what forums
            # they're most active on
            for uid in without_data[:to_go]:
                print 'pulling most active forums for user', uid
                try:
                    res = self.api.request('users.listMostActiveForums',
                                           user=uid, limit=100)
                    self.user_to_forums[uid] = [r.get('id') for r in res]
                except APIError as err:
                    if int(err.code) == 2:
                        print 'bad user id:', uid
                        self.forum_to_users[forum].remove(uid)
                    elif int(err.code) == 12:
                        # user is private: remove them from the forum's list
                        print 'user id', uid, 'is private'
                        self.forum_to_users[forum].remove(uid)
                    else:
                        return int(err.code)
                except FormattingError as err:
                    print 'Could not parse response:'
                    print err
                    self.forum_to_users[forum].remove(uid)

            print 'saving user-forum data...'
            save_json(self.user_to_forums, 'user_to_forums')

        return 0    # s'all good, man

    def pull_all_forum_users(self, min_users=1000):
        """ Loop over forums and pull in active user lists for each one """
        activity = self.get_forum_activity()

        # sort forums by 30 day activity
        forums = sorted(self.get_all_forums(), key=lambda i: -activity.get(i, 0))
        forums = [f for f in forums if f not in self.done_with]

        # loop over forums in order of most active
        for forum in forums:
            print 'pulling most active users for forum', repr(forum), 'with', \
                activity[forum], 'posts in 30 days'
            try:
                self.forum_to_users[forum] = self.pull_users(forum, min_users)
            except APIError as err:
                return int(err.code)

            print 'saving forum-user data...'
            save_json(self.forum_to_users, 'forum_to_users')
            save_json(list(self.done_with), 'done_with')
            save_json(self.all_users, 'all_users')

    def pull_forum_activity(self, forum):
        """
        Pull recent forum activity, see how many posts there are, and see if any
        of it is on threads from this year.
        """
        print 'Checking for recent posts in forum', forum
        if 'lastPost' in self.forum_details[forum] and \
                'posts30d' in self.forum_details[forum]:
            print '...already found!'
            return

        # pull most popular threads
        try:
            res = self.api.request('threads.listPopular', forum=forum,
                                   interval='30d', limit=100)
        except APIError as err:
            print err
            code = int(err.code)
            if code == 22:
                res = []
            else:
                return code
        except FormattingError as err:
            print err
            return

        # find the most recent, popular post
        last_time = datetime.datetime(2007, 1, 1)
        post_time = lambda r: dateutil.parser.parse(r['createdAt'])
        cutoff_time = datetime.datetime(2017, 1, 1)
        for r in res:
            last_time = max(last_time, post_time(r))

        self.forum_details[forum]['lastPost'] = last_time.isoformat()

        recent_threads = [r for r in res if post_time(r) > cutoff_time]
        if recent_threads:
            self.active_forum_threads[forum] = {t['id']: t for t in recent_threads}
            save_json(self.active_forum_threads[forum],
                      'active_forum_threads_' + forum)

        num_recent_posts = sum(r['postsInInterval'] for r in recent_threads)
        self.forum_details[forum]['posts30d'] = num_recent_posts
        self.forum_details[forum]['posts30dEnd'] = datetime.date.today().isoformat()

        print 'retrieved', len(recent_threads), 'threads with', \
            num_recent_posts, 'posts'
        print 'saving data...'
        save_json(self.forum_details, 'forum_details')
        print 'done.'

    def pull_forum_details(self, num_forums=-1):
        # first, get all forums that are part of our graph
        for f in self.forum_to_users.keys():
            if f not in self.forum_details:
                print 'requesting data for forum', f
                res = self.api.request('forums.details', forum=f)
                self.forum_details[f] = res
                print 'saving forum data...'
                save_json(self.forum_details, 'forum_details')

        # start getting the rest
        if not self.forum_counts:
            # this function pulls counts from the full database
            self.forum_counts = get_forum_counts(self.session)
        items = [i for i in self.forum_counts.items()
                 if i[0] not in self.forum_details]
        forums = sorted(items, key=lambda i: -i[1])

        for f, w in forums[:num_forums]:
            if f not in self.forum_details:
                print 'requesting data for forum', f, 'with', w, 'posts'
                try:
                    res = self.api.request('forums.details', forum=f)
                except APIError as err:
                    if int(err.code) == 13:
                        print 'API limit exceeded'
                        return 13
                    # Invalid argument: remove thread
                    elif int(err.code) == 2:
                        continue
                    else:
                        pdb.set_trace()
                        return err.code
                except FormattingError as err:
                    print err
                    continue

                self.forum_details[f] = res
                d = res
                print 'saving data for forum', f
                forum = Forum(pk=int(d['pk']),
                              id=d['id'],
                              name=unicode(d['name']),
                              twitter_name=unicode(d['twitterName']),
                              url=unicode(d['url']),
                              founder=int(d['founder']),
                              created_at=d['createdAt'],
                              alexa_rank=-1,
                              category=unicode(d['category']),
                              description=unicode(d['raw_description']),
                              guidelines=unicode(d['raw_guidelines']),
                              language=str(d['language']),
                              ads_enabled=bool(d['settings'].get('adsEnabled')),
                              ads_video_enabled=bool(d['settings'].get('adsVideoEnabled')),
                              adult_content=bool(d['settings'].get('adultContent')),
                              allow_anon_post=bool(d['settings'].get('allowAnonPost')),
                              allow_anon_vote=bool(d['settings'].get('allowAnonVotes')),
                              allow_media=bool(d['settings'].get('allowMedia')),
                              disable_3rd_party_trackers=bool(d['settings'].get(
                                  'disable3rdPartyTrackers')),
                              discovery_locked=bool(d['settings'].get('discoveryLocked')),
                              is_vip=bool(d['settings'].get('isVIP')),
                              must_verify=bool(d['settings'].get('mustVerify')),
                              must_verify_email=bool(d['settings'].get('mustVerifyEmail')),
                              discovery_enabled=bool(d['settings'].get(
                                  'organicDiscoveryEnabled')),
                              support_level=int(d['settings'].get(
                                  'supportLevel', -1)),
                              unapprove_links=bool(d['settings'].get(
                                  'unapproveLinks')))
                self.session.add(forum)
                self.session.commit()


        # return 0 only if we've got details on every single forum
        if num_forums > len(forums) or num_forums == -1:
            return 0

    def pull_forum_alexa_ranks(self, keyfile):
        with open(keyfile) as f:
            key_data = f.read()

        access_id = re.search('AWSAccessKeyId=(\w+)', key_data).groups()[0].strip()
        secret = re.search('AWSSecretKey=(.*)', key_data).groups()[0].strip()
        alexa_api = awis.AwisApi(access_id, secret)
        rank_key = '//{%s}Rank' % alexa_api.NS_PREFIXES['awis']

        for f, dets in self.forum_details.iteritems():
            # only pull sites with URLs, which aren't disqus channels
            if not f.startswith('channel-') and dets['url'] and \
                    'alexaRank' not in dets:
                tree = alexa_api.url_info(dets['url'], 'Rank')
                try:
                    rank = int(tree.find(rank_key).text)
                except TypeError:
                    print 'could not parse', repr(tree.find(rank_key).text), \
                        'for forum', f, 'url', dets['url']
                    continue

                print 'site', dets['url'], 'is Alexa rank', rank
                dets['alexaRank'] = rank
                save_json(self.forum_details, 'forum_details')

    def pull_all_forum_activity(self):
        weights = self.get_weights()
        items = sorted(self.forum_details.items(), key=lambda i:weights[i[0]],
                       reverse=True)
        for f, d in items:
            if 'lastPost' in d and 'posts30d' in d:
                continue

            code = self.pull_forum_activity(f)

            # check if the API limit's been reached
            if code == 13:
                return 13


    ###########################################################################
    ##  Threads and posts  ####################################################
    ###########################################################################

    def pull_thread_posts(self, thread, total_posts=1000):
        """ Pull all posts for a specified thread """

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
            res = self.api.request('threads.listPosts', thread=thread,
                                   limit=100, cursor=cursor)

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
        with open(DATA_PATH + 'threads/%s.json' % thread, 'w') as f:
            # save the thread in its own file
            json.dump(all_data, f)
        save_json(self.thread_posts, 'thread_posts')

    def pull_all_thread_posts(self, n_threads=25):
        """ Pull all posts for a certain number of threads for each forum """

        all_threads = []
        for forum, threads in self.active_forum_threads.items():
            if forum not in self.forum_details or \
                    self.forum_details[forum]['language'] != 'en':
                continue

            # only first n_threads per forum
            ts = sorted(threads.items(),
                        key=lambda i: -i[1]['postsInInterval'])[:n_threads]
            all_threads.extend([(forum, t) for i, t in ts if i not in self.thread_posts])

        # do longest threads first
        all_threads.sort(key=lambda t: -t[1]['postsInInterval'])

        # loop indefinitely, gathering data
        for forum, thread in all_threads:
            print 'pulling data for thread', repr(thread['clean_title']), \
                'from forum', thread['forum']
            try:
                self.pull_thread_posts(thread['id'])
            except APIError as err:
                print err
                # API request limit: exit
                if int(err.code) == 13:
                    print 'exiting'
                    break

                # Invalid argument: remove thread
                if int(err.code) == 2:
                    print
                    del self.active_forum_threads[forum][thread['id']]
                    save_json(self.active_forum_threads, 'active_forum_threads')

                print 'skipping thread', repr(thread['clean_title'])
            except FormattingError as err:
                print err
                print 'skipping thread', repr(thread['clean_title'])

    def pull_forum_threads(self, forum):
        """ Pull all thread metadata for a certain forum """

        if forum not in self.all_forum_threads:
            # If we haven't pulled data for this forum, start looking at the
            # first instant of President Trump's tenure
            start_time = TRUMP_START
            self.all_forum_threads[forum] = {}
            total_posts = 0
        else:
            # if we already have threads for this forum, start our query after
            # the last thread we saw
            times = [dateutil.parser.parse(d['createdAt']) for t, d in
                     self.all_forum_threads[forum].items() if t != 'complete']
            start_time = max(times)
            total_posts = len(self.all_forum_threads[forum])

        # collect data up to the end of Trump's first month
        end_time = add_month(TRUMP_START)
        last_time = start_time

        print 'pulling all threads for forum', forum

        # pull all threads in 30-day window
        cursor = None
        while last_time < end_time:
            try:
                if cursor is not None:
                    res = self.api.request('forums.listThreads', forum=forum,
                                           order='asc', limit=100,
                                           since=start_time.isoformat(),
                                           cursor=cursor)
                else:
                    res = self.api.request('forums.listThreads', forum=forum,
                                           order='asc', limit=100,
                                           since=start_time.isoformat())

            except APIError as err:
                print err
                print 'saving thread data...'
                self.all_forum_threads[forum]['complete'] = False
                save_json(self.all_forum_threads, 'all_forum_threads')
                sys.exit(1)
            except FormattingError as err:
                print err
                return

            if not res.cursor['hasNext']:
                break

            cursor = res.cursor['next']
            threads = [thread for thread in res if thread['posts'] > 0]
            num_posts = sum(t['posts'] for t in threads)
            total_posts += num_posts
            last_time = dateutil.parser.parse(res[-1]['createdAt'])
            self.all_forum_threads[forum].update({t['id']: t for t in threads})

            print "pulled %d threads with %d posts, ending on %s" % \
                (len(threads), num_posts, last_time)

        print 'retrieved', len(self.all_forum_threads[forum].keys()), \
            'threads with', total_posts, 'posts'

        print 'saving thread data...'
        self.all_forum_threads[forum]['complete'] = True
        del self.all_forum_threads[forum]['complete']
        save_json(self.all_forum_threads, 'all_forum_threads')

    def pull_all_threads(self, forums):
        """ Pull all thread metadata from a list of forums """

        print "pulling data from", len(forums), "forums"

        cutoff_time = datetime.datetime(2017, 1, 1)

        # loop indefinitely, gathering data
        for forum in forums:
            if forum in self.all_forum_threads and \
                      self.all_forum_threads[forum].get('complete', True):
                  print 'forum', forum, 'data already downloaded'
                  continue

            self.pull_forum_activity(forum)
            last_time = dateutil.parser.parse(
                self.forum_details[forum].get('lastPost'))

            if last_time < cutoff_time:
                print 'forum', forum, 'is dead since', last_time
            else:
                print 'forum', forum, 'last post on', last_time
                self.pull_forum_threads(forum)

    def pull_all_posts_window(self, forum=None, start_time=TRUMP_START,
                              stop_time=add_month(TRUMP_START)):
        """ Pull every single post made on Disqus during a certain time window """

        print 'pulling all posts ' + \
            (('from forum %s ' % forum) if forum is not None else '') + \
            'between', start_time, 'and', stop_time

        Post = get_post_db(forum=forum)
        start_ts = time.mktime(start_time.timetuple())
        stop_ts = time.mktime(stop_time.timetuple())
        cursor = None
        _, session = get_mysql_session()

        last_ts_query = session.query(func.max(Post.time)).filter(
            Post.time <= stop_time, Post.time >= start_time)

        real_min_ts = last_ts_query.first()[0]

        if real_min_ts is not None:
            start_ts = time.mktime(real_min_ts.timetuple())

        last_ts = start_ts

        if forum is not None:
            forum_obj = session.query(Forum).filter(Forum.id == forum).first()
            forum_id = forum
            forum_pk = forum_obj.pk

        # loop indefinitely, gathering posts data
        while True:
            # pull another frame of posts posts
            print 'pulling posts beginning %s...' % datetime.datetime.fromtimestamp(start_ts)
            try:
                kwargs = {'forum': ':all',
                          'limit': 100,
                          'order': 'asc',
                          'start': start_ts,
                          'end': stop_ts}

                if cursor is not None:
                    kwargs['cursor'] = cursor
                if forum is not None:
                    kwargs['forum'] = forum

                res = self.api.request('posts.list', **kwargs)

            except APIError as err:
                print err
                code = int(err.code)
                if code == 22:
                    # formatting error
                    res = []
                elif code == 8:
                    # "forum matching query does not exist": try next second
                    cursor = None
                    last_ts += 1
                    start_ts = last_ts
                    continue
                elif code == 15:
                    # "internal server error": try again
                    cursor = None
                    start_ts = last_ts
                    continue
                else:
                    return code
            except FormattingError as err:
                print err
                cursor = None
                start_ts = last_ts
                continue

            cursor = res.cursor['next']
            results = list(res)
            if not len(results):
                continue

            print 'storing %d posts between %s and %s...' % (len(results),
                                                             results[0]['createdAt'],
                                                             results[-1]['createdAt'])
            for p in results:
                post_id = int(p['id'])
                if session.query(Post).get(post_id):
                    print 'post %d already exists in database' % post_id
                    continue

                # query for forum if necessary
                if forum is None:
                    forum_id = unicode(p['forum'])
                    forum_obj = session.query(Forum).filter(Forum.id == forum_id).first()
                    if forum_obj is not None:
                        forum_pk = forum_obj.pk
                    else:
                        forum_pk = None

                # if it doesn't exist...
                post = Post(id=post_id,
                            forum=forum_id,
                            forum_pk=forum_pk,
                            thread=int(p['thread']),
                            author=int(p['author'].get('id', -1)),
                            parent=int(p['parent'] or -1),
                            raw_text=p['raw_message'],
                            time=p['createdAt'],
                            likes=int(p['likes']),
                            dislikes=int(p['dislikes']),
                            num_reports=int(p['numReports']),
                            is_approved=bool(p['isApproved']),
                            is_edited=bool(p['isEdited']),
                            is_deleted=bool(p['isDeleted']),
                            is_flagged=bool(p['isFlagged']),
                            is_spam=bool(p['isSpam']))

                try:
                    session.add(post)
                    session.commit()
                except Exception as e:
                    print
                    print e
                    print
                    pdb.set_trace()

                last_ts = time.mktime(dateutil.parser.parse(p['createdAt']).timetuple())

            print 'done.'

            # we're done if we go over time
            if last_ts > stop_ts:
                break

    ###########################################################################
    ##  Utility functions and graph stuff  ####################################
    ###########################################################################

    def get_deduped_ftu(self):
        """ Get a 'deduplicated' version of the forum_to_users dataset """
        ftu = {}
        for forum, users in self.forum_to_users.items():
            if forum in DEDUP:
                users = set(users) | set(self.forum_to_users[DEDUP[forum]])
                ftu[DEDUP[forum]] = list(users)
            else:
                ftu[forum] = users

        return ftu

    def get_deduped_utf(self):
        """ Get a 'deduplicated' version of the user_to_forums dataset """
        utf = {}
        for user, forums in self.user_to_forums.items():
            deduped = set(f for f in forums if f not in DEDUP)
            deduped |= set(DEDUP[f] for f in forums if f in DEDUP)
            utf[user] = list(deduped)

        return utf

    def get_forum_edges(self, dedup=True):
        """
        build a graph with each forum pointing to all the other forums its top
        users frequent
        """
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

            if sum(out_counts.values()):
                forum_edges[forum] = out_counts

        return forum_edges

    def get_forum_activity(self, dedup=False):
        """ map forums to recent activity """
        return {f: d.get('posts30d', 0) for f, d in self.forum_details.items()}

    def get_forum_threads(self):
        """
        return, for each forum, a list of the threads for which we have
        downloaded posts
        """
        threads = {}
        for f, ts in self.active_forum_threads.items():
            threads[f] = [t for t in ts if t in self.thread_posts]
        return threads

    def get_all_forums(self):
        """ It's in the name """
        forums = set()
        for u, fs in self.user_to_forums.items():
            forums |= set(fs)

        return list(forums)

    def get_weights(self, dedup=False):
        """ Returns a dumb, arbitrary weight for each forum

        Why is this useful, you ask? It has some vague relation to the pagerank
        of a forum -- i.e. a forum's score is based on how much it is referenced
        by other forums, and references count for more from forums with fewer
        outgoing references. This doesn't give more popular forums more outgoing
        reference weights for whatever reason. Why didn't I just use pagerank?
        good question.
        """
        forums = defaultdict(float)

        for ftf in self.get_forum_edges(dedup=dedup).values():
            for f, v in ftf.items():
                forums[f] += v / float(sum(ftf.values()))

        print 'Found', len(forums), 'forums total'

        return forums


def rank_forums(data):
    """
    When enough things are commented/uncommented in this function, it will
    print out some kind of list of important forums. I haven't used this in a
    long time and it probably doesn't work.
    """
    cutoff_time = datetime.datetime(2017, 1, 1)
    def alive_str(d):
        if 'lastPost' not in d:
            return colored('no data', 'cyan')
        if dateutil.parser.parse(d['lastPost']) > cutoff_time:
            return colored('alive', 'green')
        return colored('dead', 'red')

    forums = [(d.get('alexaRank', 10e7), f, d) for f, d in
              data.forum_details.items()]
    forums.sort()

    #activity = puller.get_forum_activity()
    #threads = puller.get_forum_threads()

    for rank, forum, details in forums:
        color = 'green' if forum in data.forum_to_users else 'red'
        print colored(forum, color), rank, alive_str(details)

        #n_users_tot = len(puller.forum_to_users.get(forum, []))
        #n_users_dl = len([u for u in puller.forum_to_users.get(forum, []) if u in
                          #puller.user_to_forums])
        #n_threads_tot = len(puller.forum_threads[forum])
        #n_threads_dl = len(threads[forum])
        #tup = (n_posts, n_threads_tot, n_users_dl, n_users_tot, n_threads_dl)
        #print colored(forum, color),
        #print '%d comments from %d threads, %d/%d active users, %d threads downloaded' % tup

    #del activity, threads


def sleep_until_next_hour():
    """ Just don't do anything until the clock strikes n+1 """
    t = datetime.datetime.now()
    next_time = datetime.datetime(t.year,t.month,t.day,(t.hour+1)%24, 1)
    print 'Sleeping until %s...' % next_time
    time.sleep((next_time - t).seconds)


if __name__ == '__main__':
    args = ap.parse_args()
    puller = DataPuller()
    DATA_PATH = args.data_path

    if args.start_time is not None:
        start_time = datetime.datetime.strptime(args.start_time, '%Y-%m-%dT%H:%M:%S')
    if args.end_time is not None:
        end_time = datetime.datetime.strptime(args.end_time, '%Y-%m-%dT%H:%M:%S')

    while True:
        start_hour = datetime.datetime.now().hour
        for kf in args.keyfile:
            puller.load_key(kf)
            code = puller.pull_all_posts_window(start_time=start_time,
                                                stop_time=end_time,
                                                forum=args.forum)
            #code = puller.pull_all_user_forums(400)
            #puller.pull_all_forum_activity()
            #code = puller.pull_forum_details(num_forums=1000)
            if code == 0:
                print "Received code 0: done!"
                sys.exit(0)
            elif code == 13:
                continue
            else:
                pdb.set_trace()

        hour = datetime.datetime.now().hour
        if code == 13 and hour == start_hour:
            # if we've hit the api limit, wait a while
            sleep_until_next_hour()


    #puller.pull_forum_details()
    #puller.pull_all_forum_activity()
    #code = puller.pull_all_forum_users()
    #puller.pull_all_tread_posts(n_threads=50)
