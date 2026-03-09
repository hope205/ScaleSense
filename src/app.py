import os
import tempfile
import streamlit as st

# --- Import your existing backend functions here ---
# from your_backend_file import run_ingestion_pipeline, initialize_agent

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

# Streamlit reruns the script on every button click. We use session_state 
# to remember the chat history and whether the database is ready.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload some CVs, name your database, and let's find your ideal candidate."}
    ]
if "db_indexed" not in st.session_state:
    st.session_state.db_indexed = False
if "agent" not in st.session_state:
    st.session_state.agent = None





with st.sidebar:
    st.header("⚙️ Database Setup")
    
    # User names the ChromaDB collection
    collection_name = st.text_input("Name this Candidate Pool (e.g., 'IT_Batch_1')", value="candidate_pool")
    
    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Upload CVs (PDFs only)", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Index Resumes", type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one CV first.")
        elif not collection_name:
            st.error("Please provide a name for the database.")
        else:
            with st.spinner("Processing and Indexing CVs... This might take a moment."):
                
                # Create a temporary directory to save the uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_paths = []
                    
                    # Save each uploaded file to the temp directory
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(temp_path)
                    
                    try:
                        # ---------------------------------------------------------
                        # 🔌 PLUG IN YOUR LlamaIndex INGESTION PIPELINE HERE
                        # Pass the 'file_paths' and 'collection_name' to your backend
                        # run_ingestion_pipeline(file_paths, collection_name)
                        # ---------------------------------------------------------
                        
                        # ---------------------------------------------------------
                        # 🔌 PLUG IN YOUR AGENT INITIALIZATION HERE
                        # st.session_state.agent = initialize_agent(collection_name)
                        # ---------------------------------------------------------
                        
                        st.session_state.db_indexed = True
                        st.success(f"Successfully indexed {len(file_paths)} resumes into '{collection_name}'!")
                        
                    except Exception as e:
                        st.error(f"An error occurred during indexing: {str(e)}")

# ==========================================
# 3. MAIN AREA: CHAT INTERFACE
# ==========================================
st.title("📄 AI Resume Screening Agent")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me to find candidates, rank resumes, or summarize experience..."):
    
    if not st.session_state.db_indexed:
        st.warning("Please upload and index resumes in the sidebar before chatting.")
    else:
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # ---------------------------------------------------------
                    # 🔌 PLUG IN YOUR LlamaIndex AGENT QUERY HERE
                    # response = st.session_state.agent.chat(prompt)
                    # full_response = str(response)
                    # ---------------------------------------------------------
                    
                    # Mock response for testing the UI before backend is connected
                    full_response = f"*(Mock Response)* I have searched the '{collection_name}' database for: '{prompt}'"
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Agent encountered an error: {str(e)}")