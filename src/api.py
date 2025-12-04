from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
from loguru import logger

from src.models import NewsArticle, ProcessedArticle, QueryRequest, QueryResponse
from src.workflow import NewsProcessingWorkflow
from src.vector_store import VectorStoreManager
from src.agents.query_agent import QueryAgent
from src.database import get_db_session
from src.models import NewsArticleDB, ExtractedEntityDB, StockImpactDB
from src.database import init_db
from src.config import config


app = FastAPI(
    title="Financial News Intelligence System",
    description="AI-powered system for processing and querying financial news",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store: VectorStoreManager = None
workflow: NewsProcessingWorkflow = None
query_agent: QueryAgent = None


@app.on_event("startup")
async def startup_event():
    global vector_store, workflow, query_agent
    logger.info("Starting Financial News Intelligence System...")
    init_db()
    vector_store = VectorStoreManager()
    workflow = NewsProcessingWorkflow(vector_store)
    query_agent = QueryAgent(vector_store)
    logger.info("System initialized successfully")


@app.get("/")
async def root():
    return {
        "message": "Financial News Intelligence System API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/ingest", response_model=ProcessedArticle)
async def ingest_article(article: NewsArticle, background_tasks: BackgroundTasks ):
    """Ingest and process news articles"""
    try:
        logger.info(f"Received article for ingestion: {article.article_id}")
        result = workflow.process_article(article)
        
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)
        
        ## response formation
        response = ProcessedArticle(
            article_id=article.article_id,
            title=article.title,
            content=article.content,
            source=article.source,
            published_at=article.published_at,
            url=article.url,
            is_duplicate=result.is_duplicate,
            duplicate_of=result.duplicate_of,
            entities=result.entities,
            stock_impacts=result.stock_impacts,
            processing_metadata=result.processing_metadata
        )
        return response
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/batch", response_model=List[ProcessedArticle])
async def ingest_batch(articles: List[NewsArticle]):
    """Ingest multiple articles"""
    try:
        logger.info(f"Received batch of {len(articles)} articles")
        results = []
        for article in articles:
            result = workflow.process_article(article)
            
            results.append(ProcessedArticle(
                article_id=article.article_id,
                title=article.title,
                content=article.content,
                source=article.source,
                published_at=article.published_at,
                url=article.url,
                is_duplicate=result.is_duplicate,
                duplicate_of=result.duplicate_of,
                entities=result.entities,
                stock_impacts=result.stock_impacts,
                processing_metadata=result.processing_metadata
            ))
        logger.info(f"Batch processing completed: {len(results)} articles")
        return results
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_news(request: QueryRequest):
    """Query news articles"""
    try:
        logger.info(f"Received query: {request.query}")
        response = query_agent.process_query(request)
        logger.info(f"Query returned {response.total_results} results")
        return response
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflow/diagram")
async def get_workflow_diagram():
    """Get workflow diagram in Mermaid format."""
    return {
        "diagram": workflow.get_graph_visualization(),
        "format": "mermaid"
    }


@app.get("/api/stats")
async def get_statistics():
    """Get system statistics."""
    try:
        with get_db_session() as session:
            total_articles = session.query(NewsArticleDB).count()
            unique_articles = session.query(NewsArticleDB).filter(
                NewsArticleDB.is_duplicate == False
            ).count()
            duplicate_articles = session.query(NewsArticleDB).filter(
                NewsArticleDB.is_duplicate == True
            ).count()
            total_entities = session.query(ExtractedEntityDB).count()
            total_impacts = session.query(StockImpactDB).count()
            
        return {
            "total_articles": total_articles,
            "unique_articles": unique_articles,
            "duplicate_articles": duplicate_articles,
            "total_entities": total_entities,
            "total_stock_impacts": total_impacts,
            "deduplication_rate": f"{(duplicate_articles / max(total_articles, 1)) * 100:.1f}%"
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_api():
    uvicorn.run(
        "src.api:app",host= config.API_HOST,port= config.API_PORT,reload=True,log_level="info"
        )

if __name__ == "__main__":
    run_api()