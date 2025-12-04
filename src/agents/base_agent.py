from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import config
from loguru import logger


class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.llm = ChatGoogleGenerativeAI(
            model= config.LLM_MODEL,
            temperature= config.LLM_TEMPERATURE
        )
        logger.info(f"Initialized {self.name}")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement process method")
    
    def log_info(self, message: str):
        """Log info message."""
        logger.info(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        """Log error message."""
        logger.error(f"[{self.name}] {message}")