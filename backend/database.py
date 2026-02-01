import os
import time
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# --- CONFIGURATION ---
# Falls back to localhost only if running outside Docker
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://recall_user:recall_password@localhost:5432/recall_db")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
VECTOR_SIZE = 512

# --- POSTGRES SETUP ---
Base = declarative_base()

class Screenshot(Base):
    __tablename__ = 'screenshots'
    
    id = Column(Integer, primary_key=True)
    filepath = Column(String, unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    app_name = Column(String, nullable=True)
    window_title = Column(String, nullable=True)
    ocr_text = Column(Text, nullable=True) 

def init_postgres():
    # Retry loop to wait for DB startup
    engine = create_engine(POSTGRES_URL)
    return sessionmaker(bind=engine)

# --- QDRANT SETUP ---
def init_qdrant():
    client = QdrantClient(url=QDRANT_URL)
    return client
