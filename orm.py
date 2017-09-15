import json
import os
from collections import defaultdict

from sqlalchemy import (create_engine, inspect, exists, Column, Unicode, String,
                        Integer, BigInteger, Boolean, DateTime, ForeignKey)
from sqlalchemy.dialects.mysql import MEDIUMTEXT, TEXT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

# try to load mysql connection info from a config file
try:
    with open('mysql-conf.json') as f:
        mysql_settings = json.load(f)
except IOError:
    print "Please supply valid MySQL connection details in mysql-conf.json."
    print "See example at mysql-conf.example.json."


# keeps track of whether we've defined each Posts table object yet
# keys are forum names (or None for the default), values are Post objects
posts_tables = {}


def get_mysql_session():
    # create MySQL database session
    engine = create_engine(URL(**mysql_settings))
    Base.metadata.create_all(bind=engine)

    Session = sessionmaker()
    Session.configure(bind=engine)
    return engine, Session()


def get_post_db(forum=None, start_time=None):
    if forum is None:
        table = 'posts_%02d_%d' % (start_time.month, start_time.year)
    else:
        table = 'posts_%s' % forum.replace('-', '_')

    if forum in posts_tables:
        return posts_tables[forum]

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
        raw_text = Column(MEDIUMTEXT(charset='utf8mb4'))
        tokens = Column(MEDIUMTEXT(charset='utf8mb4'))
        time = Column(DateTime, index=True)
        likes = Column(BigInteger)
        dislikes = Column(BigInteger)
        num_reports = Column(BigInteger)

        ## boolean attributes
        is_approved = Column(Boolean)
        is_edited = Column(Boolean)
        is_deleted = Column(Boolean)
        is_flagged = Column(Boolean)
        is_spam = Column(Boolean)

    posts_tables[forum] = Post
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
    description = Column(TEXT(charset='utf8mb4'), nullable=True)
    guidelines = Column(TEXT(charset='utf8mb4'), nullable=True)
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


class AllSidesEntry(Base):
    __tablename__ = 'allsides'

    id = Column(Integer, primary_key=True)
    name = Column(Unicode(100))
    forum_pk = Column(BigInteger, ForeignKey('forums.pk'))
    bias = Column(Integer)
    category = Column(String(50))
