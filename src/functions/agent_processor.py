from llama_index.core import PromptTemplate
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition
)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from llama_index.core import Settings

# 1. Domains
global_domains = [
    "Information Technology", "Finance", "Financial Trading", "Accounting", 
    "Human Resources", "Engineering", "Robotics", "Healthcare", "Sales", 
    "Marketing", "Operations", "Product Management", "Legal"
]
# 2. Countries
global_countries = [
    "United States", "United Kingdom", "Canada", "Australia", "India", 
    "Germany", "France", "Singapore", "Nigeria", "Remote"
]

# 3. Skills
global_skills = [
    # Software & Cloud
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "Go", "Rust", 
    "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Framer",
    "AWS", "Azure", "Google Cloud (GCP)", "Docker", "Kubernetes", "CI/CD", "Linux",
    # Data & AI
    "SQL", "NoSQL", "Machine Learning", "MLOps", "MLflow", "PyTorch", "TensorFlow", 
    "Data Engineering", "Pandas", "Computer Vision", "Large Language Models (LLMs)", "RAG",
    # Finance & Business
    "Financial Modeling", "Risk Management", "Forex Trading", "Accounting", 
    "SOX Compliance", "Excel", "Data Analysis", "Project Management", "Agile"
]
global_years_of_experience = (
    "Extract as a single integer representing the minimum years requested. "
    "For example: '5' for '5+ years', '0' for 'entry level'. Do not include text."
)


class Metadata(BaseModel):
    """
    A data model representing key professional and educational metadata extracted from a resume.
    This class captures essential candidate information including technical/professional skills, years of experience 
    based on the work experience in the resume, the professional domain they belong to,
    and the geographical distribution of their educational background.

    Attributes:
        skills (List[str]): Technical and professional competencies of the candidate
        country (List[str]): Countries where the candidate pursued formal education
        Years of Experience (int): Total years of professional experience in the relevant domain
        domain (str): The professional domain of the candidate (e.g., SALES, IT,

    Example:
        {
            "skills": ["Python", "Machine Learning", "SQL", "Project Management"],
            "country": ["United States", "India"],
            "domain": "Information Technology",
            "Years of Experience": 5,
        }
    """

    domain: str = Field(...,
                        description="The domain of the candidate can be one of SALES/ IT/ FINANCE"
                                    "Returns an empty string if no domain is identified.")

    skills: List[str] = Field(
        ...,
        description="List of technical, professional, and soft skills extracted from the resume. "
                   "and domain expertise. Returns an empty list if no skills are identified."
    )

    country: List[str] = Field(
        ...,
        description="List of countries where the candidate completed their formal education, Only extract the country."
                   "Returns an empty list if countries are not specified."
    )
    years_of_experience: int = Field(
        ...,
        description="Total years of professional experience in the relevant domain, calculated based on the work experience section of the resume. "
                   "Returns 0 if no work experience is identified."
    )







# async def get_metadata(text):
#     """Function to get the metadata from the given resume of the candidate"""
#     prompt_template = PromptTemplate("""Generate skills, and country of the education for the given candidate resume.

#     Resume of the candidate:

#     {text}""")

#     metadata = await llm.astructured_predict(
#         Metadata,
#         prompt_template,
#         text=text,
#     )

#     return metadata




def initalize_db(name):
    # 1. Initialize your ChromaDB Client and Vector Store
    db = chromadb.PersistentClient(path="../chroma_db")
    chroma_collection = db.get_or_create_collection(name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    local_embed_model = Settings.embed_model

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=local_embed_model
    )

    return index



# llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# Settings.llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct",token=os.environ.get("HUGGING_FACE_HUB_TOKEN"))



class ResumeQueryProcessor:
    def __init__(self, 
                  metadata_schema = Metadata, 
                 global_skills: list[str] = global_skills, 
                 global_countries: list[str]  = global_countries, 
                 global_domains: list[str] = global_domains, 
                 global_years_of_experience: list[str] =  global_years_of_experience,
                 db_name: str = None,
                 llm = None,
                 index = None
                 ):
        """
        Initializes the processor with the LLM and the global context strings 
        required for metadata extraction.
        """
        self.llm = llm or Settings.llm
        self.metadata_schema = metadata_schema # E.g., your Pydantic Metadata class
        self.global_skills = global_skills
        self.global_countries = global_countries
        self.global_domains = global_domains
        self.global_years_of_experience = global_years_of_experience
        self.index = initalize_db(db_name)
        

    async def get_query_metadata(self, text: str):
        """Asynchronously extracts structured metadata from a given user query."""
        prompt_template = PromptTemplate("""Generate skills, and country of the education for the given user query.

        Extracted metadata should be from the following items:

        skills: {global_skills}
        countries: {global_countries}
        domains: {global_domains}
        years of experience: {global_years_of_experience}
        user query:

        {text}""")

        extracted_metadata = await self.llm.astructured_predict(
            self.metadata_schema,
            prompt_template,
            text=text,
            global_skills=self.global_skills,
            global_countries=self.global_countries,
            global_domains=self.global_domains,
            global_years_of_experience=self.global_years_of_experience
        )

        return extracted_metadata
    
    def deduplicate_candidates_by_filepath(self, retrieved_nodes):
        """
        Takes a list of LlamaIndex NodeWithScore objects and deduplicates them 
        based on the 'file_path' metadata, keeping the highest scoring chunk.
        """
        unique_candidates = {}
        for node_with_score in retrieved_nodes:
            node = node_with_score.node
            file_path = node.metadata.get('file_path')
            # Failsafe: Skip if the chunk somehow doesn't have a file path
            if not file_path:
                continue

            # Because nodes are already sorted by score, the first time we see 
            # a file_path, it is guaranteed to be the most relevant chunk.
            if file_path not in unique_candidates:
                unique_candidates[file_path] = {
                    "file_path": file_path,
                    "domain": node.metadata.get('domain', 'N/A'),
                    "years_of_experience": node.metadata.get('years_of_experience', 0),
                    "skills": node.metadata.get('skills', ''),
                    "match_score": round(node_with_score.score, 4), 
                    # "relevant_excerpt": node.text[:400].strip() + "..."
                    "relevant_excerpt": node.text
                }

        # Return the clean dictionary values as a standard Python list
        return list(unique_candidates.values())


    async def candidates_retriever_from_query(self,query: str):
        """Synthesizes an answer to your question by feeding in an entire relevant document as context."""
        print(f"> User query string: {query}")
        # Use structured predict to infer the metadata filters and query string.
        metadata_info = await self.get_query_metadata(query)

        # 1. Start with an empty list of filters
        active_filters = []

        # 2. Dynamically check what the LLM extracted (metadata_info) and add only valid filters
        if metadata_info.domain:
            active_filters.append(
                MetadataFilter(key="domain", operator=FilterOperator.EQ, value=metadata_info.domain)
            )

        if metadata_info.skills and len(metadata_info.skills) > 0:
            active_filters.append(
                MetadataFilter(key="skills", operator=FilterOperator.IN, value=metadata_info.skills)
            )

        if metadata_info.years_of_experience and metadata_info.years_of_experience > 0:
            active_filters.append(
                MetadataFilter(key="years_of_experience", operator=FilterOperator.GTE, value=metadata_info.years_of_experience)
            )

        if metadata_info.country and len(metadata_info.country) > 0:
            active_filters.append(
                MetadataFilter(key="country", operator=FilterOperator.IN, value=metadata_info.country)
            )

        # 3. Assemble the final MetadataFilters object strictly using AND
        # If the user only asked for "Finance" and "2 years experience", 
        # active_filters will only contain those two rules.
        filters = MetadataFilters(
            filters=active_filters,
            condition=FilterCondition.OR
        )

        retriever = self.index.as_retriever(
        retrieval_mode="chunks",
        metadata_filters=filters,
        )
        # run query
        result = retriever.retrieve(query)

        # 2. Pass them through your new deduplication filter
        clean_candidates = self.deduplicate_candidates_by_filepath(result)

        print(f"> Inferred filters: {filters.model_dump_json()}")
        return clean_candidates



    # @staticmethod
    def get_candidates_file_paths(self, candidates: List[Dict[str, Any]]) -> List[str]:
        """
        Extracts unique file paths from a list of candidate dictionaries 
        (e.g., the output of our custom deduplication tool).
        """
        # Using a list comprehension inside a set is faster and cleaner!
        return list(set(candidate['file_path'] for candidate in candidates if 'file_path' in candidate))



    # @staticmethod
    def get_jd_candidates_file_paths(self, candidates: List[Any]) -> List[str]:
        """
        Extracts unique file paths from a list of LlamaIndex NodeWithScore objects
        (e.g., the raw output from a Query Engine or Retriever).
        """
        return list(set(
            candidate.node.metadata.get('file_path') 
            for candidate in candidates 
            if candidate.node.metadata.get('file_path')
        ))
    
    
    async def candidates_retriever_from_jd(self,job_description: str):
        # Use structured predict to infer the metadata filters and query string.
        metadata_info = await self.get_metadata(job_description)
        filters = MetadataFilters(
        filters=[
            MetadataFilter(key="domain", operator=FilterOperator.EQ, value=metadata_info.domain),
            MetadataFilter(key="country", operator=FilterOperator.IN, value=metadata_info.country),
            MetadataFilter(key="skills", operator=FilterOperator.IN, value=metadata_info.skills),
            MetadataFilter(key="years_of_experience", operator=FilterOperator.GTE, value=metadata_info.years_of_experience)
        ],
        condition=FilterCondition.OR
    )
        print(f"> Inferred filters: {filters.json()}")
        retriever = self.index.as_retriever(
        retrieval_mode="chunks",
        metadata_filters=filters,
        )
        # run query
        candidates_based_on_jd = retriever.retrieve(job_description)

        candidates_file_paths = self.get_jd_candidates_file_paths(candidates_based_on_jd)
        
        return candidates_file_paths

    
        
    # if __name__ == "__main__":
        # Example usage
        # processor = ResumeQueryProcessor(llm=llm, metadata_schema=YourMetadataSchema)
        # user_query = "Find me candidates with Python and AWS experience in the US with 5+ years."
        # metadata = processor.get_query_metadata(user_query)
        # print(metadata)