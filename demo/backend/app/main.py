from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import data
from app.models.session_db import Base
from app.db import engine

app = FastAPI(title="Hyper-Personalized Landing Page Backend")

# Create tables
Base.metadata.create_all(bind=engine)

app.include_router(data.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
