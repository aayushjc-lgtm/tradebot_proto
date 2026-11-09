import os
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# LlamaIndex imports for robust financial document processing
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

router = APIRouter(prefix="/api/rag", tags=["Financial RAG Pipeline"])

# Configure Global LLM and Embedding Models
Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Initialize persistent ChromaDB vector store locally
db_client = chromadb.PersistentClient(path="./chroma_financial_db")
chroma_collection = db_client.get_or_create_collection("sec_and_transcripts")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

class IngestDocumentRequest(BaseModel):
    ticker: str
    doc_type: str  # e.g., "10-K", "10-Q", "Transcript"
    period: str    # e.g., "Q3-2025"
    text_content: str

@router.post("/ingest")
async def ingest_financial_doc(payload: IngestDocumentRequest):
    """
    Ingests a raw SEC filing text or earnings transcript into the vector database.
    """
    try:
        # Create a structured LlamaIndex document with rich metadata
        doc = Document(
            text=payload.text_content,
            metadata={
                "ticker": payload.ticker.upper(),
                "doc_type": payload.doc_type,
                "period": payload.period
            },
            excluded_embed_metadata_keys=["ticker", "doc_type", "period"],
            excluded_llm_metadata_keys=["ticker", "doc_type", "period"]
        )
        
        # Build index from document (automatically chunks text and generates vectors)
        VectorStoreIndex.from_documents(
            [doc], 
            storage_context=storage_context,
            show_progress=False
        )
        
        return {
            "status": "success", 
            "message": f"Successfully indexed {payload.doc_type} for {payload.ticker} ({payload.period})"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
