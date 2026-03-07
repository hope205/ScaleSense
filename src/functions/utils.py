from llama_index.readers.file import PyMuPDFReader
from typing import List, Dict
import os

#Function to extract the text from candidates resume

def batch_extract_pdfs(file_paths: List[str]) -> Dict[str, str]:
    """
    Extracts text from a list of PDF file paths using LlamaIndex's local PyMuPDFReader.
    
    """
    # Initialize the reader once outside the loop to save memory
    reader = PyMuPDFReader()
    extracted_results = {}
    
    for file_path in file_paths:
        # 1. Check if file actually exists before trying to read it
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File not found, skipping -> {file_path}")
            continue
            
        try:
            
            # 2. Extract the data for this specific file
            documents = reader.load_data(file_path=file_path)
            
            # 3. Combine all pages into one single string
            full_text = "\n".join([doc.text for doc in documents])
            
            # 4. Save to our dictionary
            extracted_results[file_path] = full_text
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
            
    print(f"\n✅ Successfully extracted {len(extracted_results)} out of {len(file_paths)} files.")
    return extracted_results


