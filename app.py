from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.graph import build_rag_graph
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Agentic AI RAG API", 
    description="A strict-grounding RAG implementation using LangGraph and Pinecone."
)

# Compile graph on startup
try:
    graph = build_rag_graph()
    logger.info("LangGraph workflow compiled successfully.")
except Exception as e:
    logger.error(f"Failed to initialize LangGraph (check environment variables): {e}")
    graph = None

# Request / Response Schemas
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


@app.get("/health")
def health_check():
    if graph is None:
        raise HTTPException(status_code=503, detail="Application is not ready.")
    return {"status": "ok"}

@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    if graph is None:
        raise HTTPException(
            status_code=500, 
            detail="Graph workflow not initialized. Ensure OPENAI_API_KEY and PINECONE_API_KEY are valid."
        )
        
    initial_state = {
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0,
        "attempts": 0,
        "grounded": False,
    }
    
    try:
        # Invoke LangGraph workflow
        result = graph.invoke(initial_state)
        
        # Mapping response precisely to the assignment's expected JSON format
        return QueryResponse(
            query=request.query,
            final_answer=result["answer"],
            retrieved_context_chunks=result["context"],
            confidence_score=result["score"]
        )
    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))
