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
from llama_cloud.core.api_error import ApiError
from typing import List, Optional



# Load environment variables (put LLAMA_CLOUD_API_KEY in your .env file)
load_dotenv(dotenv_path=r"C:\Users\ogida\Desktop\Hope work\Tech\Vscode_files\ScaleSense\.env")

# Optionally, add your project id/organization id
llama_extract = LlamaExtract(api_key=os.getenv("LAMMA_CLOUD_API_KEY"))


agent_name = "resume-screening"






class Education(BaseModel):
    institution: str = Field(description="The institution of the candidate")
    degree: str = Field(description="The degree of the candidate")
    start_date: Optional[str] = Field(
        default=None, description="The start date of the candidate's education"
    )
    end_date: Optional[str] = Field(
        default=None, description="The end date of the candidate's education"
    )


class Experience(BaseModel):
    company: str = Field(description="The name of the company")
    title: str = Field(description="The title of the candidate")
    description: Optional[str] = Field(
        default=None, description="The description of the candidate's experience"
    )
    start_date: Optional[str] = Field(
        default=None, description="The start date of the candidate's experience"
    )
    end_date: Optional[str] = Field(
        default=None, description="The end date of the candidate's experience"
    )



class TechnicalSkills(BaseModel):
    programming_languages: List[str] = Field(
        description="The programming languages the candidate is proficient in."
    )
    frameworks: List[str] = Field(
        description="The tools/frameworks the candidate is proficient in, e.g. React, Django, PyTorch, etc."
    )
    skills: List[str] = Field(
        description="Other general skills the candidate is proficient in, e.g. Data Engineering, Machine Learning, etc."
    )


class Resume(BaseModel):
    name:  Optional[str] = Field(description="The name of the candidate")
    email:  Optional[str] = Field(description="The email address of the candidate")
    links: List[str] = Field(
        description="The links to the candidate's social media profiles"
    )
    country: Optional[str] = Field(
        default=None, description="The country the candidate is based in, if available"
    )
    experience: List[Experience] = Field(description="The candidate's experience")
    education: List[Education] = Field(description="The candidate's education")
    technical_skills: TechnicalSkills = Field(
        description="The candidate's technical skills"
    )
    key_accomplishments: str = Field(
        description="Summarize the candidates highest achievements."
    )
    years_of_experience: int = Field(
        description="The total years of experience the candidate has based on the number of years in each experience entry, if available"
    )
    domain: Optional[str] = Field(
        default=None, description="The domain the candidate has experience in, e.g. Finance, Healthcare, etc."
    )
    file_path: Optional[str] = Field(
        default=None, description="The file path of the resume that was parsed"
    )


# agent = llama_extract.create_agent(name="resume-screening", data_schema=Resume)
    
# agent.data_schema = Resume
      



def intialize_agent():
    try:
        agent = llama_extract.get_agent(name=agent_name)
        if agent:
            print(f"✅ Agent '{agent_name}' already exists. Using the existing agent.")
            pass
        else:
            agent = llama_extract.create_agent(name=agent_name, data_schema=Resume)
    except ApiError as e:
        if e.status_code == 404:
            pass
        else:
            raise

    return agent


class Extractpdf(BaseModel):
    file_paths: List[str] = Field(description="The file paths of the resumes to be extracted")
