from sqlalchemy import (create_engine, inspect, Column, String, Integer,
                        Boolean, DateTime)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Post(Base):
    __tablename__ = 'posts'
    id = Column(Integer, primary_key=True)

    # relations
    forum = Column(String(100))
    thread = Column(Integer)
    author = Column(Integer)
    parent = Column(Integer)

    # data
    raw_text = Column(String(10000))
    time = Column(DateTime)
    likes = Column(Integer)
    dislikes = Column(Integer)
    num_reports = Column(Integer)

    # boolean attributes
    is_approved = Column(Boolean)
    is_edited = Column(Boolean)
    is_deleted = Column(Boolean)
    is_flagged = Column(Boolean)
    is_spam = Column(Boolean)
