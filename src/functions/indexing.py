from dotenv import load_dotenv
from llama_cloud_services import LlamaExtract
from pydantic import BaseModel, Field
import os
import asyncio
from pathlib import Path
from llama_index.core import Document
from typing import List
from llama_index.core.prompts import PromptTemplate
from llama_index.core.async_utils import run_jobs
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
import json
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Load environment variables (put LLAMA_CLOUD_API_KEY in your .env file)
load_dotenv(dotenv_path=r"C:\Users\ogida\Desktop\Hope work\Tech\Vscode_files\ScaleSense\.env")

# Optionally, add your project id/organization id
llama_extract = LlamaExtract(api_key=os.getenv("LAMMA_CLOUD_API_KEY"))



