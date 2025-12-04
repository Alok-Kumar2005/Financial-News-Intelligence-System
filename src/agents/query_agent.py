from typing import List, Set
from datetime import datetime
from sqlalchemy import and_, or_
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.agents.base_agent import BaseAgent
from src.models import (
    QueryRequest, 
    QueryResponse, 
    ProcessedArticle, 
    Entity,
    StockImpact,
    NewsArticleDB,
    ExtractedEntityDB,
    StockImpactDB
)
from src.vector_store import VectorStoreManager
from src.agents.prompts import query_prompt
from src.database import get_db_session


class QueryAgent(BaseAgent):
    def __init__(self, vector_store: VectorStoreManager):
        super().__init__("QueryAgent")
        self.vector_store = vector_store
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", query_prompt),
            ("human", "{query}")
        ])
        
        self.parser = JsonOutputParser()
    
    def understand_query(self, query: str) -> dict:
        """ Understand query intent and extract entities."""
        try:
            chain = self.query_prompt | self.llm | self.parser
            result = chain.invoke({"query": query})
            return result
        except Exception as e:
            self.log_error(f"Query understanding failed: {e}")
            return {
                "entity_type": "general",
                "entity_name": None,
                "intent": query,
                "requires_sector_expansion": False
            }
    
    def get_article_ids_from_vector( self, query: str, max_results: int ) -> List[str]:
        """Get article IDs using semantic search."""
        docs = self.vector_store.search_articles(query, k=max_results * 2)
        return [doc.metadata["article_id"] for doc in docs]
    
    def expand_entity_context( self, entity_name: str, entity_type: str ) -> Set[str]:
        """Expand entity to related entities for context-aware retrieval."""
        expanded = {entity_name}
        
        ## if not company then adding its sector
        if entity_type == "company":
            sector_map = {
                "HDFC Bank": "Banking",
                "ICICI Bank": "Banking",
                "TCS": "Technology",
                "Infosys": "Technology",
            }
            if entity_name in sector_map:
                expanded.add(sector_map[entity_name])
        
        return expanded
    
    def retrieve_articles( self, request: QueryRequest ) -> List[ProcessedArticle]:
        """ Retrieve relevant articles based on query."""
        self.log_info(f"Processing query: {request.query}")
        # Understand the query
        query_understanding = self.understand_query(request.query)
        self.log_info(f"Query understanding: {query_understanding}")
    
        ## candidate article IDs from vector search
        article_ids = self.get_article_ids_from_vector(
            request.query,
            request.max_results
        )
        
        if not article_ids:
            return []
        
        ## getting full articles from database
        with get_db_session() as session:
            query_builder = session.query(NewsArticleDB).filter(
                NewsArticleDB.article_id.in_(article_ids),
                NewsArticleDB.is_duplicate == False
            )
            
            # Apply date filters if provided
            if request.date_from:
                query_builder = query_builder.filter(
                    NewsArticleDB.published_at >= request.date_from
                )
            if request.date_to:
                query_builder = query_builder.filter(
                    NewsArticleDB.published_at <= request.date_to
                )
            
            articles = query_builder.limit(request.max_results).all()
            
            # Build processed articles with entities and impacts
            processed_articles = []
            for article in articles:
                # Get entities
                entities_db = session.query(ExtractedEntityDB).filter(
                    ExtractedEntityDB.article_id == article.article_id
                ).all()
                
                entities = [
                    Entity(
                        text=e.entity_text,
                        type=e.entity_type,
                        confidence=e.confidence
                    )
                    for e in entities_db
                ]
                
                # Get stock impacts
                impacts_db = session.query(StockImpactDB).filter(
                    StockImpactDB.article_id == article.article_id
                ).all()
                
                impacts = [
                    StockImpact(
                        symbol=i.stock_symbol,
                        confidence=i.confidence,
                        type=i.impact_type,
                        reasoning=i.reasoning
                    )
                    for i in impacts_db
                ]
                
                processed_articles.append(ProcessedArticle(
                    article_id=article.article_id,
                    title=article.title,
                    content=article.content,
                    source=article.source,
                    published_at=article.published_at,
                    url=article.url,
                    is_duplicate=article.is_duplicate,
                    duplicate_of=article.duplicate_of,
                    entities=entities,
                    stock_impacts=impacts
                ))
        
        # If sector expansion requested, add sector-wide news
        if (request.include_sector_news and 
            query_understanding.get("requires_sector_expansion")):
            entity_name = query_understanding.get("entity_name")
            entity_type = query_understanding.get("entity_type")
            
            if entity_name and entity_type == "company":
                expanded_entities = self.expand_entity_context(
                    entity_name,
                    entity_type
                )
                self.log_info(f"Expanded entities: {expanded_entities}")
        
        return processed_articles
    
    def process_query(self, request: QueryRequest) -> QueryResponse:
        """ Process query and return response."""
        start_time = datetime.now()
        
        try:
            articles = self.retrieve_articles(request)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResponse(
                query=request.query,
                results=articles,
                total_results=len(articles),
                processing_time=processing_time,
                expanded_entities=[]
            )
            
        except Exception as e:
            self.log_error(f"Query processing failed: {e}")
            return QueryResponse(
                query=request.query,
                results=[],
                total_results=0,
                processing_time=0.0,
                expanded_entities=[]
            )