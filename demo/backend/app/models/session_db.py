from sqlalchemy import Column, Integer, String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class SessionDB(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_pseudo_id = Column(String, index=True)
    session_id = Column(Integer)
    eventtimestamp = Column(String)
    event_name = Column(String)
    transaction_id = Column(String)
    prev_event_time = Column(String)
    time_diff = Column(Float)
    new_session = Column(Boolean)
    engagement_type = Column(String)

class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)