import os
import sys
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. Debugging: Check where the library is loading from
import qdrant_client
print(f"DEBUG: Loading qdrant_client from: {qdrant_client.__file__}")

# 2. Connect
qdrant_host = os.getenv("QDRANT_URL", "http://qdrant:6333")
print(f"Connecting to: {qdrant_host}")

client = QdrantClient(url=qdrant_host)
model = SentenceTransformer('clip-ViT-B-32')

# 3. Search
query_text = "temperature and battery"
print(f"Searching for: '{query_text}'...")

vector = model.encode(query_text).tolist()

try:
    results = client.search(
        collection_name="screenshots",
        query_vector=vector,
        limit=3
    )

    for hit in results:
        print(f"\nFound ID: {hit.id} (Score: {hit.score:.4f})")
        print(f"Text: {hit.payload.get('text', '')[:100]}...")
        print(f"File: {hit.payload.get('path', 'Unknown')}")

except AttributeError as e:
    print("\nCRITICAL ERROR: The library is corrupted or shadowed.")
    print(f"Available methods: {[m for m in dir(client) if 'search' in m]}")
    raise e
