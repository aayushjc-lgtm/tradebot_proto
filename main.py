# main.py additions
from rag_engine import router as rag_router

# Include the RAG micro-service router
app.include_router(rag_router)
