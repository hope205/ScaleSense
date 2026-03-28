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
from typing import List, Dict, Any
from llama_index.core import Settings




load_dotenv()

llama_extract = LlamaExtract(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))


def _build_transformations():
    return [
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        Settings.embed_model,
    ]




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

      



def intialize_agent(agent_name):
    agent = None

    try:
        agent = llama_extract.get_agent(name=agent_name)
        if agent:
            print(f"✅ Agent '{agent_name}' already exists. Using the existing agent.")
            return agent
        # else:
        #     agent = llama_extract.create_agent(name=agent_name, data_schema=Resume)    
    except ApiError as e:
        if e.status_code == 404:
            print(f"⚠️ Agent '{agent_name}' not found. Creating a new agent.")
            agent = llama_extract.create_agent(name=agent_name, data_schema=Resume)
            
        else:
            raise

    return agent



async def initalize_db(name):
    # 1. Initialize your ChromaDB Client and Vector Store
    db = chromadb.PersistentClient(path="../chroma_db")
    chroma_collection = db.get_or_create_collection(name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    pipeline = IngestionPipeline(
        transformations=_build_transformations(),
        vector_store=vector_store,
    )


    return pipeline



class Extractpdf():
    def __init__(self, collection_name):
        self.name = collection_name
        self.agent = intialize_agent(agent_name=self.name)
        self.db = None
        

    async def initialize(self):
        # self.agent = await intialize_agent(agent_name=self.name)
        # print(self.agent)
        self.db = await initalize_db(name=self.name)

        print(f"✅ Successfully initialized agent and database for collection: '{self.name}'")


    async def batch_extract_resumes(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Scans a directory for files, queues them for asynchronous extraction via LlamaCloud,
        polls until completion, and returns the structured data as a list of dictionaries.
        
        Args:
            agent: The initialized LlamaExtract agent.
            directory_path: The local folder path containing the resumes.
            
        Returns:
            A list of dictionaries containing the extracted resume data and their file paths.
        """
        raw_resume_data = []
        resumes = []
        
        # 1. Gather all file paths in the directory
        if not os.path.exists(directory_path):
            print(f"⚠️ Directory not found: {directory_path}")
            return raw_resume_data  
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_file():
                    resumes.append(entry.path)
                    
        if not resumes:
            print(f"⚠️ No files found in {directory_path}")
            return raw_resume_data

        print(f"🚀 Queueing {len(resumes)} files for extraction...")
        
        # 2. Queue the extraction
        jobs = await self.agent.queue_extraction(resumes)

        # 3. Zip the original file paths and the resulting jobs together
        for file_path, job in zip(resumes, jobs):
            while True:
                job_status = self.agent.get_extraction_job(job.id).status
                
                if job_status == "SUCCESS":
                    break
                elif job_status in ["FAILED", "CANCELLED"]:
                    print(f"❌ Extraction failed for: {os.path.basename(file_path)}")
                    break

                await asyncio.sleep(2)
                
            # 4. Process the successful result
            if job_status == "SUCCESS":
                result = self.agent.get_extraction_run_for_job(job.id)
                
                # Safely convert from Pydantic object to a standard Python dictionary
                extracted_data = result.data
                if hasattr(extracted_data, 'model_dump'):
                    extracted_data = extracted_data.model_dump()
                elif hasattr(extracted_data, 'dict'):
                    extracted_data = extracted_data.dict() # Fallback for older Pydantic versions
                
                # Inject the file path
                extracted_data['file_path'] = file_path  
                
                # Save to our final list
                raw_resume_data.append(extracted_data)
                
        print(f"✅ Successfully extracted data for {len(raw_resume_data)} resumes.")
        return raw_resume_data


    async def ingest(self, raw_resume_data: List[Dict[str, Any]]) -> List[Resume]:
        documents_to_process = []
        for data in raw_resume_data:
            technical_skills = data.get('technical_skills', {})
            skills = technical_skills.get('skills', []) if technical_skills else []
            skills_string = ", ".join(skills)

            metadata_dict = {
                'skills': skills_string,
                'country': data.get('country', 'Unknown'),
                'domain': data.get('domain', 'Unknown'),
                'years_of_experience': data.get('years_of_experience', 0),
                'file_path': data.get('file_path', ''),
            }

            text_payload = json.dumps(data, indent=2)

            doc = Document(
                text=text_payload,
                metadata=metadata_dict,
                excluded_llm_metadata_keys=['file_path'],
            )
            documents_to_process.append(doc)

        print(f"Prepared {len(documents_to_process)} documents for ingestion.")

        # 6. Run the Pipeline
        # Note: Because this makes network calls to an API, having num_workers > 1 
        # is very helpful for speeding up the batch processing!
        nodes = self.db.run(
            documents=documents_to_process, 
            show_progress=True,
            num_workers=4 
        )

        print(f"✅ Successfully processed and saved {len(nodes)} chunks into ChromaDB!")


    def is_database_populated(self) -> bool:
        """
        Checks if the ChromaDB collection exists and contains at least one resume.
        """
        try:
            db = chromadb.PersistentClient(path="../chroma_db")
            collection = db.get_collection(name=self.name)
            
            # Check if there are any embeddings actually saved inside it
            if collection.count() > 0:
                return True
            else:
                return False
                
        except ValueError:
            # The collection does not exist at all
            return False
        except Exception as e:
            # Catch any other file permission or connection errors
            print(f"Database check error: {str(e)}")
            return False