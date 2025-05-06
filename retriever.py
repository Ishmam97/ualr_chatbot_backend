import pickle
import numpy as np
import os
import logging
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, index_path, metadata_path):
        logger.info(f"Initializing Retriever with index: {index_path}, metadata: {metadata_path}")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        try:
            self.index = faiss.read_index(index_path)
            logger.info(f"FAISS index loaded, dimension: {self.index.d}, num vectors: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            raise
        try:
            with open(metadata_path, "rb") as f:
                self.doc_metadata = pickle.load(f)
            logger.info(f"Metadata loaded, {len(self.doc_metadata)} entries")
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            raise

    def query(self, text, k=3):
        logger.info(f"Querying with text: {text}, k={k}")
        embedding = self.model.encode([text], normalize_embeddings=True)
        logger.info(f"Generated embedding shape: {embedding.shape}")
        D, I = self.index.search(np.array(embedding), k)
        logger.info(f"Search results: distances={D[0]}, indices={I[0]}")
        results = []
        for i in I[0]:
            if i < len(self.doc_metadata) and i >= 0:
                doc = self.doc_metadata[i]
                if "content" in doc:
                    results.append(doc)
                else:
                    logger.warning(f"Metadata entry {i} missing 'content' field: {doc}")
            else:
                logger.warning(f"Invalid index {i} for metadata (length: {len(self.doc_metadata)})")
        logger.info(f"Returning {len(results)} documents")
        return results