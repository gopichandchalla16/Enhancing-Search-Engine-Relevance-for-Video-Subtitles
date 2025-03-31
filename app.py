import streamlit as st
import numpy as np
import json
import assemblyai as aai
from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load AssemblyAI API key from environment
assembly_api_key = os.environ.get("ASSEMBLYAI_API_KEY")

if assembly_api_key:
    aai.settings.api_key = assembly_api_key
else:
    st.error("❌ AssemblyAI API key not found. Please provide it in your .env file.")

# ChromaDB Path
db_extract_path = os.path.join("C:/VASU_ALL_FILES/db-file-rag-project/chroma_db")

# ChromaDB Setup
client = chromadb.PersistentClient(path=db_extract_path)
collection = client.get_or_create_collection(name="subtitle_chunks")

def transcribe_audio(audio_file):
    """Transcribes audio using AssemblyAI."""
    if audio_file is None:
        return "Please upload an audio file.", None

    config = aai.TranscriptionConfig(language_code="en")
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_file)

    return transcript.text, transcript.text

def retrieve_and_display_results(query, top_n):
    """Retrieves top N subtitle search results based on query."""
    if not query:
        return "No transcription text available for search."

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    # Query ChromaDB with user-selected top_n results
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_n,
        include=["documents", "metadatas"]
    )

    return format_results_as_json(results)

def format_results_as_json(results):
    """Formats retrieved subtitle search results including subtitle text."""
    formatted_results = []

    if results and "documents" in results and results["documents"]:
        for i in range(len(results["documents"][0])):
            subtitle_text = results["documents"][0][i]  # Subtitle text (chunk)
            metadata = results["metadatas"][0][i]  # Metadata
            
            subtitle_name = metadata.get("subtitle_name", "Unknown")
            subtitle_id = metadata.get("subtitle_id", "N/A")
            url = f"https://www.opensubtitles.org/en/subtitles/{subtitle_id}"

            formatted_results.append({
                "Result": i + 1,
                "Subtitle Name": subtitle_name.upper(),
                "Subtitle Text": subtitle_text,  # Now includes actual subtitle text
                "URL": url,
            })

        return json.dumps(formatted_results, indent=4)
    
    return json.dumps([{"Result": "No results found"}], indent=4)

def clear_all():
    """Clears the transcribed text and search results."""
    st.session_state.transcribed_text = ""
    st.session_state.search_results = ""

def main():
    st.set_page_config(page_title="Shazam Clone", layout="wide")
    st.title("🎵 Shazam Clone: Audio Transcription & Subtitle Search")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        top_n_results = st.slider("Select Number of Results:", min_value=1, max_value=10, value=5)
        audio_input = st.file_uploader("📂 Upload Audio File", type=["wav", "mp3"])
        
        if st.button("🚀 Transcribe & Search"):
            if audio_input:
                transcribed_text, _ = transcribe_audio(audio_input)
                st.session_state.transcribed_text = transcribed_text
                st.session_state.search_results = retrieve_and_display_results(transcribed_text, top_n_results)
            else:
                st.warning("⚠️ Please upload an audio file first.")
    
    st.subheader("📝 Transcribed Text")
    st.text_area("", value=st.session_state.get("transcribed_text", ""), height=150)
    
    st.subheader("🔍 Subtitle Search Results")
    st.json(st.session_state.get("search_results", "{}"))
    
    if st.button("🧹 Clear All"):
        clear_all()
        st.experimental_rerun()

if __name__ == "__main__":
    main()