class QueryRAGRequest(BaseModel):
    ticker: str
    question: str

@router.post("/query")
async def query_financial_rag(payload: QueryRAGRequest):
    """
    Queries the vector database using metadata filtering for a specific ticker 
    and synthesizes an answer grounded entirely in SEC filings/transcripts.
    """
    try:
        # Load the existing vector index from storage
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        
        # Create a retriever with metadata filtering to lock results strictly to the requested ticker
        from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
        
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="ticker", value=payload.ticker.upper())]
        )
        
        retriever = index.as_retriever(
            similarity_top_k=4, 
            filters=filters
        )
        
        # Create query engine from retriever
        query_engine = index.as_query_engine(retriever=retriever)
        response = query_engine.query(payload.question)
        
        # Extract source nodes for strict auditing/citations
        sources = [
            {
                "doc_type": node.node.metadata.get("doc_type"),
                "period": node.node.metadata.get("period"),
                "snippet": node.node.text[:300] + "..."
            }
            for node in response.source_nodes
        ]
        
        return {
            "ticker": payload.ticker.upper(),
            "query": payload.question,
            "answer": str(response),
            "citations": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
