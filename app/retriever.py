# retriever.py

import pickle
import numpy as np
import os
import logging
import faiss
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, index_path, metadata_path, api_key, embedding_dim=768):
        logger.info(f"Initializing Retriever with index: {index_path}, metadata: {metadata_path}")
        self.embedding_dim = embedding_dim

        # Initialize Gemini client
        try:
            self.model = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version='v1alpha'))
            logger.info("Google Gemini embedding model initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise

        # Load FAISS index
        try:
            self.index = faiss.read_index(index_path)
            logger.info(f"FAISS index loaded, dimension: {self.index.d}, num vectors: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            raise

        # Load metadata
        try:
            with open(metadata_path, "rb") as f:
                self.doc_metadata = pickle.load(f)
            logger.info(f"Metadata loaded, {len(self.doc_metadata)} entries")
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            raise

    def query(self, text, k=3):
        logger.info(f"Querying with text: {text}, k={k}")
        try:
            response = self.model.models.embed_content(
                model='text-embedding-004',
                contents=[text],
                config=types.EmbedContentConfig(output_dimensionality=self.embedding_dim),
            )
            embedding = np.array(response.embeddings[0].values, dtype='float32').reshape(1, -1)
            logger.info(f"Generated embedding shape: {embedding.shape}")
        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            raise

        try:
            D, I = self.index.search(embedding, k)
            logger.info(f"Search results: distances={D[0]}, indices={I[0]}")
        except Exception as e:
            logger.error(f"FAISS search failed: {str(e)}")
            raise

        results = []
        for i in I[0]:
            if 0 <= i < len(self.doc_metadata):
                doc = self.doc_metadata[i]
                if "content" in doc:
                    results.append(doc)
                else:
                    logger.warning(f"Metadata entry {i} missing 'content' field: {doc}")
            else:
                logger.warning(f"Invalid index {i} for metadata (length: {len(self.doc_metadata)})")

        logger.info(f"Returning {len(results)} documents")
        return results
