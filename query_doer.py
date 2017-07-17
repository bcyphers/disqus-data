import datetime
import pdb

from orm import *
from sqlalchemy import func

TRUMP_START = datetime.datetime(2017, 1, 20, 17, 0, 0)

def get_forum_counts(session=None):
    """ get a mapping of forum ids to number of posts in the database """
    if session is None:
        _, session = get_mysql_session()
    res = session.query(Post.forum,
                        func.count(Post.id)).group_by(Post.forum).all()
    return {f: c for f, c in res}


def get_user_post_patterns(session=None):
    """
    Map users to the number of times they have posted in each forum
    """
    if session is None:
        _, session = get_mysql_session()
    res = (session.query(Post.author, Post.forum, func.count())
           .filter(Post.time >= TRUMP_START)
           .filter(Post.author != -1)
           .group_by(Post.author, Post.forum).all())
    counts = {}
    for user, forum, count in res:
        if user not in counts:
            counts[user] = {}
        counts[user][forum] = count

    return counts
