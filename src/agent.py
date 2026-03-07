from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
from pydantic import BaseModel, Field
from functions.agent_processor import ResumeQueryProcessor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.core import VectorStoreIndex






llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# 1. Initialize your ChromaDB Client and Vector Store
db = chromadb.PersistentClient(path="../chroma_db")
chroma_collection = db.get_or_create_collection("parsed_documents")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
local_embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=local_embed_model
)

processor = ResumeQueryProcessor(llm=llm, index = index)




def rank_candidates_resume(query: str) -> str:
    """
    Ranks candidates based on their resumes and the specified criteria.
    Use this tool to find the most suitable candidates for a given position.

    Args:
        query (str): The specific question to ask the database (e.g., "What are their key skills?").

    """
    
    nodes = processor.candidates_retriever_from_query(query=query)  # This will set the internal state of the processor with the retrieved candidates based on the query


    response = processor.process(query)
    
    return str(response)


resume_query_tool = FunctionTool.from_defaults(
    fn=rank_candidates_resume, 
    name="resume_query_tool", 
    description="Tool to query candidate resumes based on specific criteria"
    )


# 1. Define standard Python function with clear type hints and a docstring
def related_candidates(query: str) -> str:
    """
    Returns related candidates based on their resumes and the specified criteria given by the qery of the user.
    Use this tool to find the most suitable candidates for a given position.

    Args:
        query (str): The specific question to ask the database (e.g., "What are their key skills?").

    """
    
    nodes = processor.candidates_retriever_from_query(query=query)  
    return str(nodes)


resume_query_tool = FunctionTool.from_defaults(
    fn=related_candidates, 
    name="related_job_candidates_query_tool", 
    description="Tool to query candidate resumes based on specific criteria given by the user"
    )






query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use your tool to query a RAG system to answer information about XYZ",
    tools=[query_engine_tool],
    llm=llm
)


# ranking_agent = function_agent = FunctionAgent(
#     name="ranking_agent",           )

