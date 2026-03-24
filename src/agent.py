# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
# from llama_index.core.tools import FunctionTool
# from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
# from llama_index.core.agent.workflow import (
#     AgentWorkflow,
#     FunctionAgent,
#     ReActAgent,
# )
# from pydantic import BaseModel, Field
# from functions.agent_processor import ResumeQueryProcessor
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# import chromadb
# from llama_index.core import VectorStoreIndex
# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
# from llama_index.core.tools import FunctionTool
# from llama_index.core.agent import ReActAgent
# import asyncio
# import os
# from llama_index.core import Settings








# class ResumeScreeningAgent:
#     def __init__(self, processor):
#         """
#         Initialize the agent by passing in your custom processor and the LLM.
#         """
#         # 1. Store the processor and LLM as class attributes
#         self.processor = processor
#         self.llm = Settings.llm 
        
#         # 2. Convert your class methods into LlamaIndex tools
#         search_tool = FunctionTool.from_defaults(fn=self._search_candidates_tool)
#         ranking_tool = FunctionTool.from_defaults(fn=self._rank_candidates_resume_tool)
        

#         self.agent = AgentWorkflow.from_tools_or_functions(
#         [search_tool, ranking_tool],
#         llm=self.llm
#         )

                                                                   

#     async def _search_candidates_tool(self, query: str) -> str:
#         """
#         Searches the database for candidates matching specific criteria.
#         Use this tool whenever a user asks to find or search for candidates.
        
#         Args:
#             query (str): The search criteria (e.g., "IT professional with 5 years experience").
#         """
#         print(f"🛠️ Tool Triggered: Searching for '{query}'")
        
#         # Because we are inside the class, we have direct access to self.processor!
#         results = await self.processor.candidates_retriever_from_query(query=query)
        
#         # Format the results into a clean string for the agent to read
#         return str(results)
    
#     async def _rank_candidates_resume_tool(self, job_description: str) -> str:
#         """
#         Ranks candidates based on their resumes and the job description.
#         Use this tool to find the most suitable candidates for a given position based on the job description.

#         Args:
#             job_description (str): The specific job description.

#         """
        
#         nodes = await self.processor.candidates_retriever_from_jd(job_description)

#         # response = processor.process(query)
        
#         return str(nodes)
    

#     async def chat(self, prompt: str) -> str:
#         """
#         The main entry point for your Streamlit app to talk to the agent.
#         """
#         response = await self.agent.run(prompt)
#         return str(response)



# async def main():

#     Settings.embed_model = HuggingFaceEmbedding(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
#    )
    
#     Settings.llm  = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct",token=os.environ.get("HUGGING_FACE_HUB_TOKEN"))



#     # Example of how to initialize and test the agent independently
#     processor = ResumeQueryProcessor(db_name="test")
#     agent = ResumeScreeningAgent(processor=processor)
    
#     test_query = "Find IT professionals with 5 years of experience in Python."
#     response = await agent.chat(test_query)    
    
#     print(f"Final Response: {response}")  






# if __name__ == "__main__":
#     asyncio.run(main())





from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
from llama_index.core.workflow import Context
from pydantic import BaseModel, Field
from functions.agent_processor import ResumeQueryProcessor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.core import VectorStoreIndex, Settings
import asyncio
import os


class ResumeScreeningAgent:
    def __init__(self, processor):
        """
        Initialize the agent by passing in your custom processor and the LLM.
        """
        self.processor = processor
        self.llm = Settings.llm

        search_tool = FunctionTool.from_defaults(fn=self._search_candidates_tool)
        ranking_tool = FunctionTool.from_defaults(fn=self._rank_candidates_resume_tool)

        self.agent = AgentWorkflow.from_tools_or_functions(
            [search_tool, ranking_tool],
            llm=self.llm,
        )

    async def _search_candidates_tool(self, query: str) -> str:
        """
        Searches the database for candidates matching specific criteria.
        Use this tool whenever a user asks to find or search for candidates.

        Args:
            query (str): The search criteria (e.g., "IT professional with 5 years experience").
        """
        print(f"Tool Triggered: Searching for '{query}'")
        results = await self.processor.candidates_retriever_from_query(query=query)
        return str(results)

    async def _rank_candidates_resume_tool(self, job_description: str) -> str:
        """
        Ranks candidates based on their resumes and the job description.
        Use this tool to find the most suitable candidates for a given position
        based on the job description.

        Args:
            job_description (str): The specific job description.
        """
        nodes = await self.processor.candidates_retriever_from_jd(job_description)
        return str(nodes)

    async def chat(self, prompt: str) -> str:
        """
        The main entry point for your Streamlit app to talk to the agent.
        """
        ctx = Context(self.agent)
        response = await self.agent.run(user_msg=prompt, ctx=ctx)
        return str(response)


async def main():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Settings.llm = HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        is_chat_model=True,
    )

    processor = ResumeQueryProcessor(db_name="test")
    agent = ResumeScreeningAgent(processor=processor)

    test_query = "Find IT professionals with 5 years of experience in Python."
    response = await agent.chat(test_query)

    print(f"Final Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())