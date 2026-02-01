import os

# --- MULTICORE OPTIMIZATION ---
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import time
import logging
import yaml
from datetime import datetime
from PIL import Image

# AI Libraries
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer

# Database
# Ensure you have a database.py file in the same directory!
from database import init_postgres, init_qdrant, Screenshot, Base, VECTOR_SIZE
from qdrant_client.http.models import PointStruct, Distance, VectorParams

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Worker")

# --- CONFIG LOADING ---
# We load the same config as the watcher to ensure paths match
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Resolve Paths
INBOX_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, CONFIG.get('storage_path', '../data/inbox')))
ARCHIVE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/archive"))

# Ensure dirs exist
os.makedirs(INBOX_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

class RecallWorker:
    def __init__(self):
        logger.info("Initializing Worker...")
        logger.info(f"Watching: {INBOX_DIR}")
        logger.info(f"Archiving to: {ARCHIVE_DIR}")
        
        # 1. Wait for Databases (Retry Loop)
        self.Session = self._connect_postgres()
        self.qdrant = self._connect_qdrant()

        # 2. AI Models
        logger.info("Loading AI Models...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False) 
        self.model = SentenceTransformer('clip-ViT-B-32')
        
        logger.info("Worker Ready.")

    def _connect_postgres(self):
        while True:
            try:
                Session = init_postgres()
                engine = Session().get_bind()
                Base.metadata.create_all(engine)
                logger.info("Connected to Postgres.")
                return Session
            except Exception as e:
                logger.warning(f"Waiting for Postgres... ({e})")
                time.sleep(2)

    def _connect_qdrant(self):
        while True:
            try:
                client = init_qdrant()
                collections = client.get_collections()
                names = [c.name for c in collections.collections]
                if "screenshots" not in names:
                    client.create_collection(
                        collection_name="screenshots",
                        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                    )
                logger.info("Connected to Qdrant.")
                return client
            except Exception as e:
                logger.warning(f"Waiting for Qdrant... ({e})")
                time.sleep(2)

    def process_image(self, filename):
        filepath = os.path.join(INBOX_DIR, filename)
        
        # Safety check: Ensure file still exists
        if not os.path.exists(filepath):
            return

        try:
            # A. OCR
            # cls=True helps correct if the text is rotated
            ocr_result = self.ocr.ocr(filepath, cls=True)
            full_text = ""
            if ocr_result and ocr_result[0]:
                full_text = " ".join([line[1][0] for line in ocr_result[0]])
            
            # B. Vector Embedding
            img = Image.open(filepath)
            vector = self.model.encode(img).tolist()

            # C. Save to Postgres
            # We try to parse the timestamp from filename, else use current time
            try:
                # Filename format: 2023-10-27_10-00-00.jpeg
                # Remove extension first
                clean_name = os.path.splitext(filename)[0]
                timestamp = datetime.strptime(clean_name, "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                timestamp = datetime.now()

            session = self.Session()
            new_entry = Screenshot(
                filepath=os.path.join(ARCHIVE_DIR, filename),
                timestamp=timestamp,
                ocr_text=full_text,
                app_name="Unknown" 
            )
            session.add(new_entry)
            session.commit()
            
            # D. Save to Qdrant
            self.qdrant.upsert(
                collection_name="screenshots",
                points=[
                    PointStruct(
                        id=new_entry.id, 
                        vector=vector,
                        payload={"text": full_text, "path": filename}
                    )
                ]
            )
            session.close()

            # E. Archive File (Move from Inbox -> Archive)
            new_path = os.path.join(ARCHIVE_DIR, filename)
            os.rename(filepath, new_path)
            logger.info(f"Processed & Archived: {filename}")

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            # If duplicate in DB, delete the file to prevent infinite loop
            if "UniqueViolation" in str(e) or "IntegrityError" in str(e):
                logger.warning(f"Duplicate detected. Deleting {filename}")
                try:
                    os.remove(filepath)
                except:
                    pass

    def run(self):
        while True:
            # 1. Look for images (Supported: png, jpg, jpeg)
            # We added .jpeg here to match the new watcher
            try:
                files = sorted([f for f in os.listdir(INBOX_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            except FileNotFoundError:
                logger.warning(f"Inbox directory {INBOX_DIR} not found. Retrying...")
                time.sleep(5)
                continue
            
            if not files:
                time.sleep(2)
                continue

            for file in files:
                # 2. Skip the temporary file the watcher is currently writing to
                if file.startswith("temp_") or file == "temp_capture.jpeg":
                    continue
                
                self.process_image(file)

if __name__ == "__main__":
    worker = RecallWorker()
    worker.run()
