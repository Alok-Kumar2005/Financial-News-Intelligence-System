from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import config
from src.models import NewsArticle
from loguru import logger


class VectorStoreManager:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.news_collection = Chroma(
            client=self.client,
            collection_name="news_articles",
            embedding_function=self.embeddings,
        )
        self.entity_collection = Chroma(
            client=self.client,
            collection_name="entities",
            embedding_function=self.embeddings,
        )
        logger.info("Vector store initialized successfully")
    
    def add_article(self, article: NewsArticle, entities: List[str] = None) -> None:
        """Add article to vector store."""
        try:
            doc = Document(
                page_content=f"{article.title}\n\n{article.content}",
                metadata={
                    "article_id": article.article_id,
                    "title": article.title,
                    "source": article.source,
                    "published_at": article.published_at.isoformat(),
                    "url": article.url or "",
                }
            )
            self.news_collection.add_documents([doc], ids=[article.article_id])
            if entities:
                entity_docs = [
                    Document(
                        page_content=entity,
                        metadata={
                            "article_id": article.article_id,
                            "entity": entity
                        }
                    )
                    for entity in entities
                ]
                self.entity_collection.add_documents(
                    entity_docs,
                    ids=[f"{article.article_id}_{i}" for i in range(len(entities))]
                )
            
            logger.info(f"Added article {article.article_id} to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add article to vector store: {e}")
            raise
    
    def find_duplicates(self, article: NewsArticle, threshold: float = None) -> List[Tuple[str, float]]:
        if threshold is None:
            threshold = SIMILARITY_THRESHOLD
        
        try:
            query = f"{article.title}\n\n{article.content}"
            results = self.news_collection.similarity_search_with_score(
                query,
                k=5 
            )
            duplicates = [
                (doc.metadata["article_id"], 1.0 - score) 
                for doc, score in results
                if (1.0 - score) >= threshold and doc.metadata["article_id"] != article.article_id
            ]
            
            if duplicates:
                logger.info(f"Found {len(duplicates)} potential duplicates for article {article.article_id}")
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Failed to find duplicates: {e}")
            return []
    
    def search_articles( self, query: str, k: int = 10, filter_dict: Dict = None ) -> List[Document]:
        try:
            results = self.news_collection.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
            return results
        except Exception as e:
            logger.error(f"Failed to search articles: {e}")
            return []
    
    def search_by_entity(self, entity: str, k: int = 10) -> List[str]:
        try:
            results = self.entity_collection.similarity_search(
                entity,
                k=k
            )
            return [doc.metadata["article_id"] for doc in results]
        except Exception as e:
            logger.error(f"Failed to search by entity: {e}")
            return []
    
    def delete_article(self, article_id: str) -> None:
        try:
            self.news_collection.delete(ids=[article_id])
            logger.info(f"Deleted article {article_id} from vector store")
        except Exception as e:
            logger.error(f"Failed to delete article: {e}")
    
    def reset(self) -> None:
        try:
            self.client.reset()
            logger.warning("Vector store reset")
        except Exception as e:
            logger.error(f"Failed to reset vector store: {e}")