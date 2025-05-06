from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import Retriever
from llm import call_gemini
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="UALR Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update paths for containerized environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "ualr_chatbot_backend","faiss_index.index")
METADATA_PATH = os.path.join(BASE_DIR, "ualr_chatbot_backend", "doc_metadata.pkl")

logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"INDEX_PATH: {INDEX_PATH}")
logger.info(f"METADATA_PATH: {METADATA_PATH}")

try:
    retriever = Retriever(index_path=INDEX_PATH, metadata_path=METADATA_PATH)
except Exception as e:
    logger.error(f"Failed to initialize Retriever: {str(e)}")
    raise RuntimeError(f"Retriever initialization failed: {str(e)}")

class QueryRequest(BaseModel):
    query: str
    api_key: str
    k: int = 3
    model: str = "gemini-1.5-flash-latest"

SYSTEM_PROMPT = """
You are a helpful chatbot for the University of Arkansas at Little Rock (UALR). 
Use the following context to answer the question concisely and accurately. 
If the context is empty or lacks specific details, respond with: 
"I was unable to find specific information regarding this, but here is what you can do: 
Contact UALR's main office at (501) 569-3000 or email info@ualr.edu for further assistance."
"""

@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query}, k: {request.k}")

        docs = retriever.query(request.query, k=request.k)
        context = "\n".join([doc.get("content", "") for doc in docs])
        logger.info(f"Retrieved {len(docs)} documents, context length: {len(context)}")

        prompt = f"Context:\n{context}\n\nQuestion: {request.query}\n\nAnswer concisely:"
        
        response = call_gemini(
            api_key=request.api_key,
            prompt=prompt,
            model=request.model,
            system_prompt=SYSTEM_PROMPT
        )
        
        return {"response": response, "retrieved_docs": docs}
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "UALR Chatbot API is running"}

@app.get("/")
async def root():
    return {"message": "UALR Chatbot API is running"}