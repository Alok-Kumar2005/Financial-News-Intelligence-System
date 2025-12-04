import os
import sys
import asyncio
from typing import Union
from pydantic import BaseModel
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from src.load_config import load_config
from src.logger import logging
from src.exception import CustomException
from dotenv import load_dotenv

load_dotenv()

class LLMChainFactory:
    """Factory class to build LangChain LLM chains for Google Gemini and Groq models."""
    def __init__(self, model_type: str = "gemini"):
        self.model_type = model_type
        self._load_environment()
        self._load_config()

    def _load_environment(self):
        """Loads environment variables required for LLMs."""
        load_dotenv()
        # os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
        # os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "")
        # os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

    def _load_config(self):
        """Loads the LLM configuration from YAML."""
        try:
            logging.info("Loading LLM configuration...")
            config = load_config()
            llm_config = config["llm"]
            self.gemini_model_name = llm_config["google"]["model_name"]
            self.gemini_model_kwargs = llm_config["google"].get("model_kwargs", {})
            self.groq_model_name = llm_config["groq"]["model_name"]
            self.groq_model_kwargs = llm_config["groq"].get("model_kwargs", {})

        except Exception as e:
            logging.error(f"Failed to load config: {str(e)}")
            raise CustomException(e, sys)

    def _get_llm(self):
        """Initializes and returns the correct LLM based on model_type."""
        try:
            if self.model_type == "gemini":
                logging.info("Initializing Google Gemini LLM...")
                return ChatGoogleGenerativeAI(
                    model=self.gemini_model_name,
                    google_api_key=self.google_api_key,
                    **self.gemini_model_kwargs
                )

            elif self.model_type == "groq":
                logging.info("Initializing Groq LLM...")
                return ChatGroq(
                    model=self.groq_model_name,
                    api_key=self.groq_api_key,
                    **self.groq_model_kwargs
                )

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        except Exception as e:
            logging.error(f"Error initializing LLM: {str(e)}")
            raise CustomException(e, sys)

    async def get_llm_async(self):
        """Asynchronously returns an LLM instance."""
        return self._get_llm()

    async def get_llm_chain(self, prompt: Union[PromptTemplate, ChatPromptTemplate]):
        """Creates and returns an async LLM chain with a prompt."""
        try:
            logging.info("Creating LLM chain...")
            llm = await self.get_llm_async()
            chain = prompt | llm
            return chain
        except Exception as e:
            logging.error(f"Error creating LLM chain: {str(e)}")
            raise CustomException(e, sys)

    async def get_structured_llm_chain(self, prompt: Union[PromptTemplate, ChatPromptTemplate], output_schema: BaseModel):
        """Creates a structured output chain using a Pydantic model."""
        try:
            logging.info("Creating structured LLM chain...")
            llm = await self.get_llm_async()
            structured_llm = llm.with_structured_output(output_schema)
            chain = prompt | structured_llm
            return chain
        except Exception as e:
            logging.error(f"Error in structured LLM chain: {str(e)}")
            raise CustomException(e, sys)

    async def get_llm_tool_chain(self, prompt: Union[PromptTemplate, ChatPromptTemplate], tools: list):
        """Creates an async LLM chain integrated with tools."""
        try:
            logging.info("Creating tool-integrated LLM chain...")
            llm = await self.get_llm_async()
            llm_with_tools = llm.bind_tools(tools)
            chain = prompt | llm_with_tools
            return chain
        except Exception as e:
            logging.error(f"Error creating tool LLM chain: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    async def test_async():
        factory = LLMChainFactory(model_type="gemini")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{input}")
        ])
        
        chain = await factory.get_llm_chain(prompt)
        response = await chain.ainvoke({"input": "What is the capital of France?"})
        print(response.content)

    asyncio.run(test_async())