from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional, List, Dict, Any
import re
from langsmith import Client

from retriever import Retriever
from llm import call_gemini

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
BASE_DIR = "/app"
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "doc_metadata.pkl")

LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
if not LANGSMITH_API_KEY:
    logger.warning("LangSmith API key not found. Feedback submission to LangSmith will be disabled.")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "ualr-chatbot")
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_FILE = os.path.join(APP_DIR, "feedback_log.jsonl")

# Add this log line to see the exact path being used when the app starts
logger.info(f"Attempting to use feedback log file at: {FEEDBACK_FILE}")
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"INDEX_PATH: {INDEX_PATH}")
logger.info(f"METADATA_PATH: {METADATA_PATH}")

langsmith_client = None
if LANGSMITH_API_KEY:
    try:
        langsmith_client = Client(
            api_key=LANGSMITH_API_KEY,
            api_url=LANGSMITH_ENDPOINT
        )
        logger.info(f"LangSmith client initialized with project: {LANGSMITH_PROJECT}")
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith client: {e}")


def extract_uuid_from_run_id(run_id: str) -> str:
    """
    Extract UUID from LangChain run_id format.
    Examples:
    - "run--9f67587f-11c2-4a3f-aef1-1b57a8d5a31d-0" -> "9f67587f-11c2-4a3f-aef1-1b57a8d5a31d"
    - "9f67587f-11c2-4a3f-aef1-1b57a8d5a31d" -> "9f67587f-11c2-4a3f-aef1-1b57a8d5a31d"
    """
    if not run_id:
        return run_id
    
    # Try to extract UUID pattern from the run_id
    uuid_pattern = r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
    match = re.search(uuid_pattern, run_id, re.IGNORECASE)
    
    if match:
        extracted_uuid = match.group(1)
        logger.info(f"Extracted UUID '{extracted_uuid}' from run_id '{run_id}'")
        return extracted_uuid
    
    # If no UUID pattern found, return original (might already be a clean UUID)
    logger.warning(f"Could not extract UUID from run_id: {run_id}")
    return run_id

class FeedbackItem(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            # Tell Pydantic how to encode datetime objects
            datetime: lambda dt: dt.isoformat()
        }
    )
    
    timestamp: datetime
    query: Optional[str] = None
    response: Optional[str] = None
    feedback_type: str  # e.g., "thumbs_up", "thumbs_down", "correction_suggestion"
    thumbs_down_reason: Optional[str] = None
    thumbs_up_reason: Optional[str] = None
    corrected_question: Optional[str] = None
    correct_answer: Optional[str] = None
    model_used: Optional[str] = None
    retrieved_docs: Optional[List[Dict[str, Any]]] = None
    source_message_id: Optional[str] = None
    run_id: Optional[str] = None 

class QueryRequest(BaseModel):
    query: str
    api_key: str
    k: int = 3
    model: str = "gemini-1.5-flash-latest"

SYSTEM_PROMPT = """
You are a helpful chatbot for the University of Arkansas at Little Rock (UALR). 
Use the following context to answer the question. 
Be helpful with your answer by describing what can be done to resolve the question,
add ths to your response if you do not know the answer: 
"I was unable to find specific information regarding this, but here is what you can do: 
Contact UALR's main office at (501) 569-3000 or email info@ualr.edu for further assistance."
"""


@app.post("/feedback")
async def store_feedback(feedback: FeedbackItem):
    # More verbose logging for this specific function
    request_timestamp_str = feedback.timestamp.isoformat()  # Get a string representation for logging
    
    # Print the raw request data for debugging
    logger.info(f"Received raw feedback data: {feedback}")
    logger.info(f"feedback_type: {feedback.feedback_type}")
    
    # Log specific fields based on feedback type
    if feedback.feedback_type == "thumbs_up":
        logger.info(f"thumbs_up_reason: {feedback.thumbs_up_reason}")
    elif feedback.feedback_type == "thumbs_down":
        logger.info(f"thumbs_down_reason: {feedback.thumbs_down_reason}")
    elif feedback.feedback_type == "correction_suggestion":
        logger.info(f"corrected_question: {feedback.corrected_question}")
        logger.info(f"correct_answer: {feedback.correct_answer}")
    
    logger.info(f"Target feedback file for this request: {FEEDBACK_FILE}")

    if feedback.run_id and langsmith_client:
        try:
            # Extract clean UUID from run_id
            clean_run_id = extract_uuid_from_run_id(feedback.run_id)
            
            score = 1.0 if feedback.feedback_type == "thumbs_up" else 0.0
            comment = feedback.thumbs_up_reason if feedback.feedback_type == "thumbs_up" else feedback.thumbs_down_reason
            if feedback.feedback_type == "correction_suggestion":
                comment = f"Correction: Q: {feedback.corrected_question}, A: {feedback.correct_answer}"
                score = 0.0

            logger.info(f"Submitting feedback to LangSmith with run_id: {clean_run_id}")
            langsmith_client.create_feedback(
                run_id=clean_run_id,
                key="user_rating",
                score=score,
                comment=comment or "No comment",
                # Optional: specify project if needed
                # project_id=LANGSMITH_PROJECT
            )
            logger.info(f"Feedback submitted to LangSmith for run_id {clean_run_id}")
        except Exception as e:
            logger.error(f"Failed to submit feedback to LangSmith: {e}")

    try:
        logger.info(f"[{request_timestamp_str}] Attempting to open file '{FEEDBACK_FILE}' for append...")
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            logger.info(f"[{request_timestamp_str}] File '{FEEDBACK_FILE}' opened successfully in append mode.")
            
            # Convert model to JSON string
            json_data = feedback.model_dump_json()
            logger.info(f"[{request_timestamp_str}] JSON data to write: {json_data}")
            
            line_to_write = json_data + "\n"
            logger.info(f"[{request_timestamp_str}] Attempting to write line (length {len(line_to_write)})")
            
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
        retriever= Retriever(
            index_path=INDEX_PATH,
            metadata_path=METADATA_PATH,
            api_key=request.api_key
        )
        logger.info(f"Retriever initialized with index: {INDEX_PATH}, metadata: {METADATA_PATH}")
        docs = retriever.query(request.query, k=request.k)
        context = "\n".join([doc.get("content", "") for doc in docs])
        logger.info(f"Retrieved {len(docs)} documents, context length: {len(context)}")

        prompt = f"Question: {request.query}\n\nContext:\n{context}\n\nAnswer:"
        
        response = call_gemini(
            api_key=request.api_key,
            prompt=prompt,
            model=request.model,
            system_prompt=SYSTEM_PROMPT
        )

        # Log the response ID for debugging
        logger.info(f"LangChain response ID: {response.id}")
        
        return {"response": response, "retrieved_docs": docs, "run_id": response.id}
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "UALR Chatbot API is running"}

@app.get("/")
async def root():
    return {"message": "UALR Chatbot API is running"}
