import os
import tempfile
import asyncio
import re

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from functions.indexing import Extractpdf
from functions.agent_processor import ResumeQueryProcessor
from agent import ResumeScreeningAgent

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
Settings.llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    is_chat_model=True,
)


def sanitize_collection_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip())
    return slug.strip("_-") or "default"



@st.cache_resource
def get_or_create_agent(collection_name: str):
    """
    Initializes the agent and keeps its network client alive across reruns.
    """
    try:
        extractor = Extractpdf(collection_name=collection_name)
        
        # Check if DB is empty before initializing the heavy agent
        if not extractor.is_database_populated():
            return None, "Database is empty."
            
        # Initialize the agent
        asyncio.run(extractor.initialize())
        
        # Return the fully initialized agent
        return extractor.agent, "Success"
        
    except Exception as e:
        return None, str(e)





st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

# ==========================================
# 1. INITIALIZE SESSION STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Enter your agent's name to connect to an existing database, or upload new CVs to get started."}
    ]
if "db_indexed" not in st.session_state:
    st.session_state.db_indexed = False
if "agent" not in st.session_state:
    st.session_state.agent = None


if "cv_agent" not in st.session_state:
    st.session_state.cv_agent = None

if "query_processor" not in st.session_state:
    st.session_state.query_processor = None

# ==========================================
# 2. SIDEBAR: CONNECT & UPLOAD
# ==========================================
with st.sidebar:
    st.header("⚙️ Agent Setup")
    
    # 1. User names the Agent / Database
    collection_name = st.text_input("Agent / Database Name", value="IT_Candidates")

    # --- ACTION 1: Connect to Existing Database ---
    if st.button("Connect to Agent"):
        if not collection_name:
            st.error("Please provide an agent name.")
        else:
            with st.spinner(f"Connecting to {collection_name}..."):
                safe_name = sanitize_collection_name(collection_name)
                extractor = Extractpdf(collection_name=safe_name)
                processor = ResumeQueryProcessor(db_name=safe_name)

                st.session_state.cv_agent = ResumeScreeningAgent(processor=processor)


                
                # 2. STRICT CHECK: Does this database actually have resumes?
                if not extractor.is_database_populated():
                    st.session_state.db_indexed = False
                    st.error(f"❌ No records found for '{collection_name}'. Please check the name or upload new resumes below.")
                else:
                    try:
                        # 3. Only initialize the heavy LlamaIndex agent if the DB is valid
                        asyncio.run(extractor.initialize())
                        
                        st.session_state.agent = extractor.agent
                        st.session_state.db_indexed = True
                        st.success(f"✅ Connected to '{collection_name}'! Ready to chat.")
                        
                    except Exception as e:
                        st.session_state.db_indexed = False
                        st.error(f"Could not initialize the agent: {str(e)}")






    st.divider()
    
    # --- ACTION 2: Upload New Resumes ---
    st.subheader("Add New Resumes")
    uploaded_files = st.file_uploader(
        "Upload CVs (PDFs only)", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Upload & Index", type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one CV first.")
        elif not collection_name:
            st.error("Please provide a name for the database.")
        else:
            with st.spinner("Processing and Indexing CVs... This might take a moment."):
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_paths = []
                    
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(temp_path)
                    
                    try:
                        safe_name = sanitize_collection_name(collection_name)
                        extractor = Extractpdf(collection_name=safe_name)
                        asyncio.run(extractor.initialize())
                        
                        # Run extraction and ingestion
                        resume_data = asyncio.run(extractor.batch_extract_resumes(directory_path=temp_dir))
                        asyncio.run(extractor.ingest(raw_resume_data=resume_data))

                        # Save agent to session state and unlock chat
                        st.session_state.agent = extractor.agent
                        st.session_state.db_indexed = True
                        st.success(f"Successfully indexed {len(file_paths)} resumes into '{collection_name}'!")
                        
                    except Exception as e:
                        st.error(f"An error occurred during indexing: {str(e)}")



# ==========================================
# 3. MAIN AREA: CHAT INTERFACE
# ==========================================
st.title("📄 AI Resume Screening Agent")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me to find candidates, rank resumes, or summarize experience..."):
    
    # Block chat if they haven't connected or indexed successfully
    if not st.session_state.db_indexed:
        st.warning("Please connect to an existing agent or upload resumes in the sidebar before chatting.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(st.session_state.cv_agent.chat(prompt))
                    full_response = str(response)
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Agent encountered an error: {str(e)}")



