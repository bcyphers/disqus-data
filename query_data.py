from datetime import datetime, timedelta

from orm import *
from sqlalchemy import func
import plotly.graph_objs as go
import plotly.offline as py
import numpy as np

TRUMP_START = datetime(2017, 1, 20, 17, 0, 0)


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
    Get the number of posts per day for a single forum
    'forum' must be one of the forums with its own table
    """
    if session is None:
        _, session = get_mysql_session()
    Post = get_post_db(forum=forum)
    date_func = func.date_format(Post.time, '%Y-%m-%d')
    res = session.query(date_func, func.count(Post.id)) \
           .group_by(date_func).all()

    results = {datetime.strptime(d, '%Y-%m-%d'): c for d, c in res}
    day = datetime.strptime(res[0][0], '%Y-%m-%d')
    end_day = datetime.strptime(res[-1][0], '%Y-%m-%d')
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


def get_user_post_patterns(session=None):
    """
    Map users to the number of times they have posted in each forum
    """
    if session is None:
        _, session = get_mysql_session()
    res = session.query(Post.author, Post.forum, func.count()) \
           .filter(Post.time >= TRUMP_START) \
           .filter(Post.author != -1) \
           .group_by(Post.author, Post.forum).all()
    counts = {}
    for user, forum, count in res:
        if user not in counts:
            counts[user] = {}
        counts[user][forum] = count

    return counts

def tokenize_posts(forum):
    """
    Convert the raw_text for every one of a forum's comments to list of tokens.
    Save as 'tokens' column in database.
    This function processes 30 days of posts at a time.
    """
    if session is None:
        _, session = get_mysql_session()

    Post = get_post_db(forum)
    engine, session = get_mysql_session()
    tokenize = StemTokenizer(stem=False)

    # find the time of the first post we have for this forum that doesn't have
    # tokens
    window_start = session.query(func.min(Post.time))\
        .filter(Post.tokens == None).first()[0]
    end_time = session.query(func.max(Post.time))\
        .filter(Post.tokens == None).first()[0]

    while window_start < end_time:
        window_end = min(window_start + timedelta(days=30), end_time)
        print "Tokenizing posts between %s and %s" % (window_start, window_end)

        # query for all forum posts in our time window
        query = session.query(Post)\
            .filter(Post.time >= window_start and Post.time < window_end)\
            .update(Post.tokens: tokenize(Post.raw_text))
        session.commit()

        #post_df = pd.read_sql(query.statement, query.session.bind)
        #tokens = post_df.raw_text.apply(tokenize)

        window_start += timedelta(days=30)

