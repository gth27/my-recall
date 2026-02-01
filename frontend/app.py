import shutil
import time
import datetime
import streamlit as st
import os
import pandas as pd
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(page_title="My Recall", layout="wide")

# Database Connections
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://recall_user:recall_password@postgres:5432/recall_db")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

# --- CACHED RESOURCES ---
@st.cache_resource
def get_db_engine():
    return create_engine(POSTGRES_URL)

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)

@st.cache_resource
def get_model():
    return SentenceTransformer('clip-ViT-B-32')

# Initialize
try:
    engine = get_db_engine()
    qdrant = get_qdrant_client()
    model = get_model()
except Exception as e:
    st.error(f"Connection Error: {e}")
    st.stop()

# --- SIDEBAR: CONTROLS & FILTERS ---
st.sidebar.title("My Recall")
PAUSE_FILE = "/data/recall.pause"

with st.sidebar:
    # 1. SEARCH SETTINGS
    st.subheader("Search Settings")
    search_mode = st.radio(
        "Search Mode",
        ("Hybrid (Recommended)", "Text Only (Exact)", "Visual Only (AI)"),
        help="Text Only is faster. Visual finds concepts."
    )
    
    # 2. TIME TRAVEL (New Feature)
    st.divider()
    st.subheader("Time Travel")
    
    # Default to "All Time" (no filter)
    use_date_filter = st.checkbox("Filter by Date")
    start_date, end_date = None, None
    
    if use_date_filter:
        today = datetime.date.today()
        # Default to last 7 days
        date_range = st.date_input(
            "Select Range",
            (today - datetime.timedelta(days=7), today),
            format="MM/DD/YYYY"
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
            # Ensure end_date covers the whole day (until 23:59:59)
            # We handle this logic in the SQL params below

    st.divider()

    # 3. STATUS & CONTROLS
    st.subheader("System Status")
    is_paused = os.path.exists(PAUSE_FILE)
    
    if is_paused:
        st.error("Capture Paused")
        if st.button("Resume Capture"):
            if os.path.exists(PAUSE_FILE):
                os.remove(PAUSE_FILE)
            st.rerun()
    else:
        st.success("Capture Running")
        if st.button("Pause Capture"):
            with open(PAUSE_FILE, "w") as f:
                f.write("paused")
            st.rerun()

    st.divider()

    # 4. DANGER ZONE
    if st.button("Delete All Data", type="primary"):
        with st.spinner("Nuking everything..."):
            if not is_paused:
                with open(PAUSE_FILE, "w") as f:
                    f.write("paused")
            
            try:
                with engine.connect() as conn:
                    conn.execute(text("TRUNCATE TABLE screenshots"))
                    conn.commit()
                qdrant.delete_collection("screenshots")
                qdrant.create_collection("screenshots", vectors_config=VectorParams(size=512, distance=Distance.COSINE))
                
                folders = ["/data/archive", "/data/inbox"]
                for folder in folders:
                    if os.path.exists(folder):
                        for filename in os.listdir(folder):
                            file_path = os.path.join(folder, filename)
                            if os.path.isfile(file_path): os.unlink(file_path)
            except Exception as e:
                st.error(f"Reset failed: {e}")

        st.success("Memory Wiped!")
        time.sleep(1)
        st.rerun()

# --- MAIN SEARCH ---
st.title("Search your history")
query = st.text_input("What are you looking for?", placeholder="e.g. 'cpu usage' or 'youtube video about cats'")

# --- BUILD DATE FILTERS ---
# We append these conditions to our SQL queries
date_clause = ""
params = {"q": f"%{query}%"}

if use_date_filter and start_date and end_date:
    date_clause = " AND timestamp >= :start AND timestamp <= :end"
    # Make end date inclusive (end of that day)
    params["start"] = start_date
    params["end"] = end_date + datetime.timedelta(days=1)

# --- SEARCH LOGIC ---
results = {}

if query:
    with st.spinner(f"Searching ({search_mode})..."):
        
        # 1. TEXT SEARCH (SQL)
        if search_mode in ["Hybrid (Recommended)", "Text Only (Exact)"]:
            sql = f"SELECT id, filepath, ocr_text FROM screenshots WHERE ocr_text ILIKE :q {date_clause} ORDER BY timestamp DESC LIMIT 20"
            try:
                with engine.connect() as conn:
                    df_text = pd.read_sql(text(sql), conn, params=params)
                
                for _, row in df_text.iterrows():
                    results[row['id']] = {
                        "id": row['id'], "score": 1.0, "path": os.path.basename(row['filepath']),
                        "text": row['ocr_text'], "type": "Text Match"
                    }
            except Exception as e:
                st.warning(f"Text search error: {e}")

        # 2. AI SEARCH (Vector -> SQL Validation)
        if search_mode in ["Hybrid (Recommended)", "Visual Only (AI)"]:
            try:
                # A. Get Candidates from AI
                vector = model.encode(query).tolist()
                ai_hits = qdrant.search(collection_name="screenshots", query_vector=vector, limit=30)
                
                if ai_hits:
                    # B. Validate Dates via SQL (since Qdrant doesn't have the timestamp index)
                    ai_ids = [hit.id for hit in ai_hits]
                    
                    # If we have a date filter, we MUST check the DB. 
                    # If no date filter, we can skip this check if we trust Qdrant, 
                    # but checking DB is safer to ensure file still exists.
                    if use_date_filter:
                        sql_chk = f"SELECT id FROM screenshots WHERE id IN :ids {date_clause}"
                        with engine.connect() as conn:
                            valid_df = pd.read_sql(text(sql_chk), conn, params={**params, "ids": tuple(ai_ids)})
                            valid_ids = set(valid_df['id'].tolist())
                    else:
                        valid_ids = set(ai_ids) # Assume all valid if no date filter

                    # C. Add valid results
                    for hit in ai_hits:
                        if hit.id in valid_ids and hit.id not in results:
                            results[hit.id] = {
                                "id": hit.id, "score": hit.score,
                                "path": hit.payload.get("path"), "text": hit.payload.get("text", ""),
                                "type": "AI Match"
                            }
            except Exception as e:
                st.warning(f"AI search error: {e}")

else:
    # 3. NO QUERY (Show Recent)
    # We still respect the date filter here!
    recency_clause = f"WHERE 1=1 {date_clause}" if use_date_filter else ""
    sql = f"SELECT id, filepath, timestamp, ocr_text FROM screenshots {recency_clause} ORDER BY timestamp DESC LIMIT 12"
    
    # We need to pass params if date filter is on
    query_params = params if use_date_filter else {}
    
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=query_params)
    
    for _, row in df.iterrows():
        results[row['id']] = {
            "id": row['id'], "score": None, "path": os.path.basename(row['filepath']),
            "text": row['ocr_text'], "type": "Recent"
        }

# --- DISPLAY ---
final_results = sorted(list(results.values()), key=lambda x: x['score'] if x['score'] else 0, reverse=True)

if final_results:
    cols = st.columns(3)
    for idx, item in enumerate(final_results):
        col = cols[idx % 3]
        full_path = os.path.join("/data/archive", item['path'])
        
        with col:
            try:
                # Use standard PIL image opening
                image = Image.open(full_path)
                st.image(image, use_container_width=True)
                
                if item['type'] == "🔍 Text Match":
                    st.success(f"Text Match")
                elif item.get('score'):
                    st.caption(f"Confidence: {item['score']:.2f}")
                else:
                    st.caption(f"Timeline")

                with st.expander("Details"):
                    st.text(item['text'][:300])
                    
            except Exception:
                st.error("Image not found")
else:
    if use_date_filter:
        st.info("No memories found in this time range.")
    else:
        st.info("No memories found.")
