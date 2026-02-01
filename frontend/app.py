import shutil
import time
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

# --- SIDEBAR: CONTROLS & STATS ---
st.sidebar.title("Recall Memory")
PAUSE_FILE = "/data/recall.pause"

with st.sidebar:
    # 1. SEARCH SETTINGS
    st.subheader("Search Settings")
    search_mode = st.radio(
        "Search Mode",
        ("Hybrid (Recommended)", "Text Only (Exact)", "Visual Only (AI)"),
        help="Text Only is faster and precise. Visual finds concepts (e.g. 'cat')."
    )
    
    st.divider()

    # 2. STATUS & CONTROLS
    st.subheader("Status")
    is_paused = os.path.exists(PAUSE_FILE)
    
    if is_paused:
        st.error("⏸️ Capture Paused")
        if st.button("▶️ Resume Capture"):
            if os.path.exists(PAUSE_FILE):
                os.remove(PAUSE_FILE)
            st.rerun()
    else:
        st.success("🟢 Capture Running")
        if st.button("⏸️ Pause Capture"):
            with open(PAUSE_FILE, "w") as f:
                f.write("paused")
            st.rerun()

    st.divider()

    # 3. STATS
    try:
        df_count = pd.read_sql("SELECT COUNT(*) FROM screenshots", engine)
        count = df_count.iloc[0, 0]
        st.metric("Total Memories", count)
    except:
        st.metric("Status", "Database Offline")

    st.divider()

    # 4. DANGER ZONE
    st.subheader("Danger Zone")
    if st.button("Delete All Data", type="primary"):
        with st.spinner("Nuking everything..."):
            if not is_paused:
                with open(PAUSE_FILE, "w") as f:
                    f.write("paused")
            
            # Clear Database
            try:
                with engine.connect() as conn:
                    conn.execute(text("TRUNCATE TABLE screenshots"))
                    conn.commit()
            except Exception as e:
                st.error(f"SQL Error: {e}")

            # Clear Vectors
            try:
                qdrant.delete_collection("screenshots")
                qdrant.create_collection(
                    collection_name="screenshots",
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                )
            except Exception as e:
                st.warning(f"Vector reset warning: {e}")

            # Delete Files
            folders = ["/data/archive", "/data/inbox"]
            for folder in folders:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        file_path = os.path.join(folder, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            st.error(f"Failed to delete {file_path}. Reason: {e}")

        st.success("Memory Wiped Successfully!")
        time.sleep(2)
        st.rerun()

# --- MAIN SEARCH ---
st.title("Search your history")
query = st.text_input("What are you looking for?", placeholder="e.g. 'cpu usage' or 'youtube video about cats'")

# --- SMART SEARCH LOGIC ---
results = {}

if query:
    with st.spinner(f"Searching using {search_mode}..."):
        
        # 1. TEXT SEARCH (Run if Hybrid OR Text Only)
        if search_mode in ["Hybrid (Recommended)", "Text Only (Exact)"]:
            sql_query = text("SELECT id, filepath, ocr_text FROM screenshots WHERE ocr_text ILIKE :q ORDER BY id DESC LIMIT 20")
            try:
                with engine.connect() as conn:
                    df_text = pd.read_sql(sql_query, conn, params={"q": f"%{query}%"})
                    
                for _, row in df_text.iterrows():
                    filename = os.path.basename(row['filepath'])
                    results[row['id']] = {
                        "id": row['id'],
                        "score": 1.0,
                        "path": filename,
                        "text": row['ocr_text'],
                        "type": "Text Match"
                    }
            except Exception as e:
                st.warning(f"Text search failed: {e}")

        # 2. AI SEARCH (Run if Hybrid OR Visual Only)
        if search_mode in ["Hybrid (Recommended)", "Visual Only (AI)"]:
            try:
                vector = model.encode(query).tolist()
                search_result = qdrant.search(
                    collection_name="screenshots",
                    query_vector=vector,
                    limit=20
                )
                
                for hit in search_result:
                    # If existing text match, skip (Text takes priority)
                    if hit.id not in results:
                        results[hit.id] = {
                            "id": hit.id,
                            "score": hit.score,
                            "path": hit.payload.get("path"),
                            "text": hit.payload.get("text", ""),
                            "type": "AI Match"
                        }
            except Exception as e:
                st.warning(f"AI search failed: {e}")

else:
    # Default: Show latest images
    query_sql = "SELECT id, filepath, timestamp, ocr_text FROM screenshots ORDER BY id DESC LIMIT 12"
    df = pd.read_sql(query_sql, engine)
    for _, row in df.iterrows():
        filename = os.path.basename(row['filepath'])
        results[row['id']] = {
            "id": row['id'],
            "score": None,
            "path": filename,
            "text": row['ocr_text'],
            "type": "Recent"
        }

# --- DISPLAY GRID ---
final_results = sorted(list(results.values()), key=lambda x: x['score'] if x['score'] else 0, reverse=True)

if final_results:
    cols = st.columns(3)
    
    for idx, item in enumerate(final_results):
        col = cols[idx % 3]
        full_path = os.path.join("/data/archive", item['path'])
        
        with col:
            try:
                image = Image.open(full_path)
                st.image(image, use_container_width=True)
                
                if item['type'] == "Text Match":
                    st.success(f"Matched Text: \"{query}\"")
                elif item.get('score'):
                    st.caption(f"AI Confidence: {item['score']:.2f}")
                
                with st.expander("See Text"):
                    st.text(item['text'][:500])
                    
            except FileNotFoundError:
                st.error(f"File missing: {item['path']}")
else:
    st.info("No memories found.")
