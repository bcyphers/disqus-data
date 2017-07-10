from orm import *
from sqlalchemy import func

def get_forum_counts(session):
    """ get a mapping of forum ids to number of posts in the database """
    res = session.query(Post.forum,
                        func.count(Post.id)).group_by(Post.forum).all()
    return {f: c for f, c in res}
