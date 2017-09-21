from datetime import datetime, timedelta
from collections import defaultdict

from sqlalchemy import func, update, select
from sqlalchemy import Table, Column, Integer, String, MetaData
import plotly.graph_objs as go
import plotly.offline as py
import numpy as np
import pandas as pd

from orm import get_post_db, get_mysql_session
from load_text import StemTokenizer
from constants import TRUMP_START, DATE_FMT


def get_forum_posts_count(year=2017, session=None):
    """ get a mapping of forum ids to number of posts in the database """
    if session is None:
        _, session = get_mysql_session()
    Post = get_post_db(start_time=datetime(year, 1, 1))
    res = session.query(Post.forum, func.count(Post.id)) \
            .group_by(Post.forum).all()
    return {f: c for f, c in res}


def get_forum_posts_per_day(forum, session=None):
    """
    Get the number of posts per day for a single forum.
    'forum' must be one of the forums with its own table.
    """
    if session is None:
        _, session = get_mysql_session()
    Post = get_post_db(forum=forum)
    date_func = func.date_format(Post.time, DATE_FMT)
    res = session.query(date_func, func.count(Post.id)) \
           .group_by(date_func).all()

    results = {datetime.strptime(d, DATE_FMT): c for d, c in res}
    day = datetime.strptime(res[0][0], DATE_FMT)
    end_day = datetime.strptime(res[-1][0], DATE_FMT)
    while day < end_day:
        day += timedelta(days=1)
        if day not in results:
            results[day] = 0

    return results


def rolling_avg(a, n):
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret /= n
    return ret[n-1:]


def plot_posts_per_day(forum, window=7):
    days = sorted(get_forum_posts_per_day(forum).items())
    x, y = zip(*days)
    ny = rolling_avg(y, window)
    nx = x[window-1:]

    scatter = go.Scatter(x=x, y=y, mode='markers')
    line = go.Scatter(x=nx, y=ny)
    py.plot([scatter, line], filename='%s-activity.html' % forum)


def get_user_forums(year=2017, cutoff=2):
    """
    Map users to the number of times they have posted in each forum
    """
    engine, _ = get_mysql_session()

    metadata = MetaData()
    table = Table('author_forums_%d' % year, metadata,
                  Column('author', Integer),
                  Column('forum', String),
                  Column('count(*)', Integer))
    conn = engine.connect()
    result = conn.execute(select([table]))

    user_docs = defaultdict(list)
    user_counts = defaultdict(int)
    # we're gonna treat each user as a document lol
    for user, forum, count in result:
        user_docs[user] += (int(count) * [str(forum)])
        user_counts[user] += 1

    for user, count in user_counts.iteritems():
        if count < cutoff:
            del user_docs[user]

    return user_docs


def get_user_threads(year=2017):
    """
    Map users to the number of times they have posted in each thread
    """
    engine, _ = get_mysql_session()

    Post = get_post_db(start_time=datetime(year, 1, 1))
    res = session.query(Post.author, Post.thread, func.count()) \
           .filter(Post.time >= TRUMP_START) \
           .filter(Post.author != -1) \
           .group_by(Post.author, Post.thread).all()
    counts = {}
    for user, forum, count in res:
        if user not in counts:
            counts[user] = {}
        counts[user][forum] = count

    return counts


def tokenize_posts(forum, start_time=None, end_time=None):
    """
    Convert the raw_text for every one of a forum's comments to list of tokens.
    Save as 'tokens' column in database.
    This function processes 30 days of posts at a time.
    """
    _, session = get_mysql_session()

    Post = get_post_db(forum)
    posts = Post.__table__
    engine, session = get_mysql_session()
    tokenize = StemTokenizer(stem=False)

    if start_time is None:
        # find the time of the first post for this forum that doesn't have tokens
        print 'querying for first post...'
        start_time = session.query(func.min(Post.time))\
            .filter(Post.tokens == None).first()[0]
    if end_time is None:
        print 'querying for last post...'
        end_time = session.query(func.max(Post.time))\
            .filter(Post.tokens == None).first()[0]

    window_start = start_time
    while window_start < end_time:
        window_end = min(window_start + timedelta(days=30), end_time)
        print "querying for posts from %s between %s and %s" % (forum, window_start, window_end)

        # query for all forum posts in our time window
        query = session.query(Post)\
            .filter(Post.time >= window_start)\
            .filter(Post.time < window_end)\
            .all()
        print "found %d posts for %s, tokenizing..." % (len(query), forum)

        # tokenize each post and update the 'tokens' column
        checkpoint = datetime.now()
        for i, p in enumerate(query):
            p.tokens = ' '.join(tokenize(p.raw_text))
            if datetime.now() - checkpoint > timedelta(seconds=30):
                print "tokenized %d posts" % (i + 1)
                checkpoint = datetime.now()

        print "committing update..."
        session.commit()
        window_start += timedelta(days=30)

