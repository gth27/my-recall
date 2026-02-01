import os

# --- MULTICORE OPTIMIZATION ---
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import time
import logging
from datetime import datetime
from PIL import Image

# AI Libraries
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer

# Database
from database import init_postgres, init_qdrant, Screenshot, Base, VECTOR_SIZE
from sqlalchemy import create_engine
from qdrant_client.http.models import PointStruct, Distance, VectorParams

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIG
INBOX_DIR = "/data/inbox"
ARCHIVE_DIR = "/data/archive"
os.makedirs(ARCHIVE_DIR, exist_ok=True)

class RecallWorker:
    def __init__(self):
        logger.info("Initializing Worker...")
        
        # 1. Wait for Databases (Retry Loop)
        self.Session = self._connect_postgres()
        self.qdrant = self._connect_qdrant()

        # 2. AI Models
        logger.info("Loading AI Models... (May take time on first run)")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False) 
        self.model = SentenceTransformer('clip-ViT-B-32')
        
        logger.info("Worker Ready. Watching inbox...")

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
        
        # Safety check: Ensure file still exists (race condition protection)
        if not os.path.exists(filepath):
            return

        try:
            # A. OCR
            ocr_result = self.ocr.ocr(filepath, cls=True)
            full_text = ""
            if ocr_result and ocr_result[0]:
                full_text = " ".join([line[1][0] for line in ocr_result[0]])
            
            # B. Vector
            img = Image.open(filepath)
            vector = self.model.encode(img).tolist()

            # C. Save
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

            # D. Move file
            new_path = os.path.join(ARCHIVE_DIR, filename)
            # Use shutil.move or replace to handle overwrites if needed, 
            # but os.rename is atomic and safer here.
            os.rename(filepath, new_path)
            logger.info(f"Processed: {filename}")

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            # If it's a unique constraint error, we might want to move/delete the file 
            # so it doesn't loop forever.
            if "UniqueViolation" in str(e):
                logger.warning(f"Duplicate detected. Deleting duplicate file: {filename}")
                try:
                    os.remove(filepath)
                except:
                    pass

    def run(self):
        while True:
            files = sorted([f for f in os.listdir(INBOX_DIR) if f.endswith(('.png', '.jpg'))])
            
            if not files:
                time.sleep(2)
                continue

            for file in files:
                if file.startswith("temp_") or file == "temp_capture.png":
                    continue
                
                self.process_image(file)

if __name__ == "__main__":
    worker = RecallWorker()
    worker.run()
