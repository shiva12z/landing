from fastapi import APIRouter, HTTPException, Depends, status, Request
from typing import List
from app.models.session import Session
from app.services.ml import recommend_engagement_type, segment_users, cold_start_recommendation, personalize_landing_page
from sqlalchemy.orm import Session as OrmSession
from fastapi import Depends
from app.db import get_db
from app.models.session_db import SessionDB
from app.models.user import UserDB, UserCreate, Base as UserBase
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
import pandas as pd
import os
import json
from passlib.hash import bcrypt

router = APIRouter(prefix="/data", tags=["data"])

@router.get("/sessions")
def get_sessions():
    if not os.path.exists("preprocessed_sessions.csv"):
        raise HTTPException(status_code=404, detail="Preprocessed data not found.")
    try:
        df = pd.read_csv("preprocessed_sessions.csv", nrows=100)
        # Convert to JSON-compliant output
        data = json.loads(df.to_json(orient="records"))
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{user_pseudo_id}")
def get_user_sessions(user_pseudo_id: str):
    if not os.path.exists("preprocessed_sessions.csv"):
        raise HTTPException(status_code=404, detail="Preprocessed data not found.")
    try:
        df = pd.read_csv("preprocessed_sessions.csv")
        user_sessions = df[df['user_pseudo_id'] == user_pseudo_id]
        if user_sessions.empty:
            raise HTTPException(status_code=404, detail=f"No sessions found for user {user_pseudo_id}")
        data = json.loads(user_sessions.to_json(orient="records"))
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendation/{user_pseudo_id}")
def get_user_recommendation(user_pseudo_id: str):
    result = recommend_engagement_type(user_pseudo_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No recommendation available for user {user_pseudo_id}")
    return {"user_pseudo_id": user_pseudo_id, "recommended_engagement_type": result}

@router.get("/segments")
def get_user_segments():
    segments = segment_users()
    if not segments:
        raise HTTPException(status_code=404, detail="Segmentation data not available.")
    return segments

@router.post("/coldstart")
def cold_start(user_profile: dict = None):
    """Recommend engagement type for a new user using cold start strategy."""
    result = cold_start_recommendation(user_profile or {})
    return {"recommended_engagement_type": result}

@router.get("/personalize/{user_pseudo_id}")
def personalize_user(user_pseudo_id: str):
    """Return personalized landing page content for a known user."""
    content = personalize_landing_page(user_pseudo_id=user_pseudo_id)
    return content

@router.post("/personalize_coldstart")
def personalize_coldstart(user_profile: dict = None):
    """Return personalized landing page content for a new user (cold start)."""
    content = personalize_landing_page(user_profile=user_profile or {})
    return content

@router.get("/sessions_db")
def get_sessions_db(db: OrmSession = Depends(get_db)):
    sessions = db.query(SessionDB).limit(100).all()
    return [s.__dict__ for s in sessions]

@router.get("/sessions_db_example")
def get_sessions_db_example(db: OrmSession = Depends(get_db)):
    """Example endpoint: returns up to 10 sessions from the SQLite database."""
    sessions = db.query(SessionDB).limit(10).all()
    return [
        {
            "user_pseudo_id": s.user_pseudo_id,
            "session_id": s.session_id,
            "eventtimestamp": s.eventtimestamp,
            "event_name": s.event_name,
            "transaction_id": s.transaction_id,
            "prev_event_time": s.prev_event_time,
            "time_diff": s.time_diff,
            "new_session": s.new_session,
            "engagement_type": s.engagement_type,
        }
        for s in sessions
    ]

@router.post("/login")
def login_user(user: UserCreate, db: OrmSession = Depends(get_db), request: Request = None):
    print(f"[LOGIN DEBUG] Path: {request.url.path}, Method: {request.method}, Body: {user}")
    db_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if not db_user or not bcrypt.verify(user.password, db_user.password):
        print("[LOGIN DEBUG] Invalid credentials")
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    print("[LOGIN DEBUG] Login successful")
    return {"message": "Login successful."}

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: OrmSession = Depends(get_db), request: Request = None):
    print(f"[REGISTER DEBUG] Path: {request.url.path}, Method: {request.method}, Body: {user}")
    if db.query(UserDB).filter(UserDB.email == user.email).first():
        print("[REGISTER DEBUG] Email already registered")
        raise HTTPException(status_code=400, detail="Email already registered.")
    hashed_password = bcrypt.hash(user.password)
    db_user = UserDB(email=user.email, password=hashed_password)
    db.add(db_user)
    try:
        db.commit()
        db.refresh(db_user)
    except IntegrityError:
        db.rollback()
        print("[REGISTER DEBUG] IntegrityError: Email already registered")
        raise HTTPException(status_code=400, detail="Email already registered.")
    print("[REGISTER DEBUG] Registration successful")
    return {"message": "User registered successfully."}
