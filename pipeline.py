import os
import pickle
import numpy as np
import faiss
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types

# ---- CONFIG ----
DATA_DIR = "./data"
INDEX_PATH = "./faiss_index.faiss"
METADATA_PATH = "./doc_metadata.pkl"
EMBEDDING_DIM = 768  # use 768 as per your retriever
CHUNK_SIZE = 500     # chars per chunk (tune as needed)
CHUNK_OVERLAP = 100  # chars overlap for continuity
API_KEY = os.environ.get("GEMINI_API_KEY")  # or set directly

def chunk_text(text, chunk_size=500, overlap=100):
    """Yields chunks of text with optional overlap."""
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        yield text[start:end]
        start += chunk_size - overlap

def read_txt(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def read_xlsx(path):
    dfs = pd.read_excel(path, sheet_name=None)
    chunks = []
    for sheet, df in dfs.items():
        for _, row in df.iterrows():
            line = " | ".join(str(cell) for cell in row if not pd.isna(cell))
            if line.strip():
                chunks.extend(chunk_text(line))
    return chunks

def embed_batch(texts, client):
    # batching is important for memory
    resp = client.models.embed_content(
        model="text-embedding-004",
        contents=list(texts),
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    # returns a response with .embeddings list
    arr = [np.array(e.values, dtype="float32") for e in resp.embeddings]
    return arr

def main():
    assert API_KEY, "Set GOOGLE_API_KEY env var (or fill API_KEY directly)."
    client = genai.Client(api_key=API_KEY, http_options=types.HttpOptions(api_version='v1alpha'))

    all_chunks = []
    all_metadata = []
    # Collect text chunks and their metadata
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if fname.endswith(".txt"):
            text = read_txt(fpath)
            for chunk in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source_file": fname,
                    "content": chunk
                })
        elif fname.endswith(".xlsx"):
            xlsx_chunks = read_xlsx(fpath)
            for chunk in xlsx_chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "source_file": fname,
                    "content": chunk
                })
        # add other formats as needed

    # Memory-saving batching for embeddings
    BATCH_SIZE = 32
    embeddings = []
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Embedding"):
        batch = all_chunks[i:i+BATCH_SIZE]
        try:
            batch_emb = embed_batch(batch, client)
        except Exception as e:
            print(f"Embedding batch {i} failed: {e}")
            continue
        embeddings.extend(batch_emb)

    # Build FAISS index
    emb_arr = np.stack(embeddings)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(emb_arr)
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved FAISS index to {INDEX_PATH}")

    # Save metadata (for retriever)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(all_metadata, f)
    print(f"Saved metadata ({len(all_metadata)} entries) to {METADATA_PATH}")

if __name__ == "__main__":
    main()
