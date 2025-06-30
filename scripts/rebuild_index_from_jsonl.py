import json
import pickle
import numpy as np
import faiss
from google import genai
from google.genai import types
import os

def embed_texts(texts, api_key, embedding_dim=768):
    print(f"🔍 Generating embeddings for {len(texts)} texts...")
    BATCH_SIZE = 100
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        print(f"Embedding batch {i} → {i+len(batch)} of {len(texts)}")
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=batch,
            config=types.EmbedContentConfig(output_dimensionality=embedding_dim),
        )
        batch_vectors = [np.array(embed.values, dtype="float32") for embed in response.embeddings]
        embeddings.extend(batch_vectors)
    return embeddings

def rebuild_index(jsonl_path, index_path, metadata_path, api_key):
    print(f"Loading JSONL: {jsonl_path}")
    with open(jsonl_path, "r") as f:
        texts = [json.loads(line)["text"] for line in f]

    embeddings = embed_texts(texts, api_key)
    dim = len(embeddings[0])
    print(f"Vector dimension: {dim}")
    
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")

    metadata = [{"content": text} for text in texts]
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to: {metadata_path}")
    print(f"Indexed {len(texts)} documents total.")

if __name__ == "__main__":
    rebuild_index(
        jsonl_path="notebook/program_grad_coord.jsonl",
        index_path="faiss_index.faiss",
        metadata_path="doc_metadata.pkl",
        api_key=os.getenv("GOOGLE_API_KEY") 
    )
