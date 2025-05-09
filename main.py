from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import Retriever
from llm import call_gemini
import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

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
INDEX_PATH = os.path.join(BASE_DIR, "backend", "ualr_chatbot", "faiss_index.index")
METADATA_PATH = os.path.join(BASE_DIR, "backend", "ualr_chatbot", "doc_metadata.pkl")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_FILE = os.path.join(APP_DIR, "feedback_log.jsonl")
# Add this log line to see the exact path being used when the app starts
logger.info(f"Attempting to use feedback log file at: {FEEDBACK_FILE}")

logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"INDEX_PATH: {INDEX_PATH}")
logger.info(f"METADATA_PATH: {METADATA_PATH}")

try:
    retriever = Retriever(index_path=INDEX_PATH, metadata_path=METADATA_PATH)
except Exception as e:
    logger.error(f"Failed to initialize Retriever: {str(e)}")
    raise RuntimeError(f"Retriever initialization failed: {str(e)}")

#models
class FeedbackItem(BaseModel):
    timestamp: datetime
    query: Optional[str] = None
    response: Optional[str] = None
    feedback_type: str # e.g., "thumbs_up", "thumbs_down", "correction_suggestion"
    thumbs_down_reason: Optional[str] = None
    corrected_question: Optional[str] = None
    correct_answer: Optional[str] = None
    model_used: Optional[str] = None
    retrieved_docs: Optional[List[Dict[str, Any]]] = None 

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

@app.post("/feedback")
async def store_feedback(feedback: FeedbackItem):
    # More verbose logging for this specific function
    request_timestamp_str = feedback.timestamp.isoformat() # Get a string representation for logging
    logger.info(f"--- FEEDBACK START for timestamp: {request_timestamp_str} ---")
    logger.info(f"Received feedback data: {feedback.model_dump_json(indent=2)}")
    logger.info(f"Target feedback file for this request: {FEEDBACK_FILE}")

    try:
        logger.info(f"[{request_timestamp_str}] Attempting to open file '{FEEDBACK_FILE}' for append...")
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f: # Added encoding
            logger.info(f"[{request_timestamp_str}] File '{FEEDBACK_FILE}' opened successfully in append mode.")
            
            line_to_write = feedback.model_dump_json() + "\n"
            logger.info(f"[{request_timestamp_str}] Attempting to write line (length {len(line_to_write)}): {line_to_write.strip()}") # Log line without trailing newline
            
            f.write(line_to_write)
            logger.info(f"[{request_timestamp_str}] f.write() called. Line should be in file buffer.")
            
            f.flush()  # Explicitly flush the OS buffer to disk
            logger.info(f"[{request_timestamp_str}] f.flush() called. OS buffer flushed to disk.")
            
        # The 'with' block automatically closes the file here.
        logger.info(f"[{request_timestamp_str}] File '{FEEDBACK_FILE}' closed. Write operation complete for this request.")
        logger.info(f"--- FEEDBACK END for timestamp: {request_timestamp_str} ---")
        return {"status": "success", "message": "Feedback received"}
    
    except Exception as e:
        # Log the full traceback for any exception
        logger.error(f"!!! CRITICAL ERROR storing feedback for timestamp {request_timestamp_str} to {FEEDBACK_FILE}: {str(e)}", exc_info=True)
        logger.info(f"--- FEEDBACK END (WITH ERROR) for timestamp: {request_timestamp_str} ---")
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {str(e)}")


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