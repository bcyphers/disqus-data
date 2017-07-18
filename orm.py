import os
from sqlalchemy import (create_engine, inspect, exists, Column, Unicode, String,
                        Integer, BigInteger, Boolean, DateTime)
from sqlalchemy.dialects.mysql import MEDIUMTEXT, TEXT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

# MySQL database info
LOCAL_MYSQL_DB = {
    'drivername': 'mysql',
    'username': 'bcyphers',
    'host': 'localhost',
    'port': 3306,
    'password': os.environ['MYSQL_PASSWD'],
    'database': 'disqus',
    'query': {'charset': 'utf8mb4'},
}

REMOTE_MYSQL_DB = {
    'drivername': 'mysql',
    'username': 'bcyphers',
    'host': '34.210.178.215',
    'port': 3306,
    'password': os.environ['MYSQL_PASSWD'],
    'database': 'disqus',
    'query': {'charset': 'utf8mb4'},
}


def get_mysql_session(remote=True):
    # create MySQL database session
    if remote:
        engine = create_engine(URL(**REMOTE_MYSQL_DB))
    else:
        engine = create_engine(URL(**LOCAL_MYSQL_DB))

    Base.metadata.create_all(bind=engine)

    Session = sessionmaker()
    Session.configure(bind=engine)
    return engine, Session()


def get_post_db(forum=None):
    if forum is None:
        table = 'posts'
    else:
        table = 'posts_%s' % forum.replace('-', '_')

    class Post(Base):
        __tablename__ = table

        id = Column(BigInteger, primary_key=True, autoincrement=False)

        ## relations
        # this is here so that forums which arent in the "forums" table yet can be
        # pulled down later
        forum = Column(Unicode(255))
        forum_pk = Column(BigInteger)
        thread = Column(BigInteger)
        author = Column(BigInteger)
        parent = Column(BigInteger)

        ## data
        raw_text = Column(MEDIUMTEXT(unicode=True))
        time = Column(DateTime)
        likes = Column(BigInteger)
        dislikes = Column(BigInteger)
        num_reports = Column(BigInteger)

        ## boolean attributes
        is_approved = Column(Boolean)
        is_edited = Column(Boolean)
        is_deleted = Column(Boolean)
        is_flagged = Column(Boolean)
        is_spam = Column(Boolean)

    return Post


class Forum(Base):
    __tablename__ = 'forums'

    # identifiers
    pk = Column(BigInteger, primary_key=True, autoincrement=False)
    id = Column(Unicode(255))
    name = Column(Unicode(255))
    twitter_name = Column(Unicode(255))
    url = Column(Unicode(255))

    # relations
    founder = Column(BigInteger)

    # metadata
    created_at = Column(DateTime)
    alexa_rank = Column(BigInteger)
    category = Column(Unicode(255), nullable=True)
    description = Column(TEXT(unicode=True), nullable=True)
    guidelines = Column(TEXT(unicode=True), nullable=True)
    language = Column(String(10))

    # settings
    ads_enabled = Column(Boolean)
    ads_video_enabled = Column(Boolean)
    adult_content = Column(Boolean)
    allow_anon_post = Column(Boolean)
    allow_anon_vote = Column(Boolean)
    allow_media = Column(Boolean)
    disable_3rd_party_trackers = Column(Boolean)
    discovery_locked = Column(Boolean)
    is_vip = Column(Boolean)
    must_verify = Column(Boolean)
    must_verify_email = Column(Boolean)
    discovery_enabled = Column(Boolean)
    support_level = Column(Integer)
    unapprove_links = Column(Boolean)
