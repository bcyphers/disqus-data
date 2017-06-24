from sqlalchemy import (create_engine, inspect, Column, Unicode, BigInteger,
                        Boolean, DateTime)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Post(Base):
    __tablename__ = 'posts'
    id = Column(BigInteger, primary_key=True, autoincrement=False)

    # relations
    forum = Column(Unicode(100))
    thread = Column(BigInteger)
    author = Column(BigInteger)
    parent = Column(BigInteger)

    # data
    raw_text = Column(Unicode(10000))
    time = Column(DateTime)
    likes = Column(BigInteger)
    dislikes = Column(BigInteger)
    num_reports = Column(BigInteger)

    # boolean attributes
    is_approved = Column(Boolean)
    is_edited = Column(Boolean)
    is_deleted = Column(Boolean)
    is_flagged = Column(Boolean)
    is_spam = Column(Boolean)
