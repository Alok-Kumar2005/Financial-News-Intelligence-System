from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class NewsArticleDB(Base):
    ## database model for news articels
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String, nullable=False)
    published_at = Column(DateTime, nullable=False)
    url = Column(String, nullable=True)
    is_duplicate = Column(Boolean, default=False)
    duplicate_of = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ExtractedEntityDB(Base):
    ## models for news entity extraction
    __tablename__ = "extracted_entities"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(String, index=True, nullable=False)
    entity_text = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class StockImpactDB(Base):
    ## model for stock impact information
    __tablename__ = "stock_impacts"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(String, index=True, nullable=False)
    stock_symbol = Column(String, index=True, nullable=False)
    confidence = Column(Float, nullable=False)
    impact_type = Column(String, nullable=False)
    reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Entity(BaseModel):
    ## entity from news article
    text: str
    type: str
    confidence: float = 1.0


class StockImpact(BaseModel):
    ## stock impact info
    symbol: str
    confidence: float
    type: str
    reasoning: Optional[str] = None


class NewsArticle(BaseModel):
    ## news article model
    article_id: str
    title: str
    content: str
    source: str
    published_at: datetime
    url: Optional[str] = None


class ProcessedArticle(BaseModel):
    article_id: str
    title: str
    content: str
    source: str
    published_at: datetime
    url: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    entities: List[Entity] = []
    stock_impacts: List[StockImpact] = []
    processing_metadata: Dict = {}


class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    max_results: int = Field(default=10, ge=1, le=100)
    include_sector_news: bool = True
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


class QueryResponse(BaseModel):
    """Query response model."""
    query: str
    results: List[ProcessedArticle]
    total_results: int
    processing_time: float
    expanded_entities: List[str] = []


# Agent State Models
class AgentState(BaseModel):
    """State passed between agents in LangGraph."""
    article: NewsArticle
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    entities: List[Entity] = []
    stock_impacts: List[StockImpact] = []
    processing_metadata: Dict = {}
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True