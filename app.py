import streamlit as st
import numpy as np
import json
import assemblyai as aai
from sentence_transformers import SentenceTransformer
import chromadb
import backoff
import requests
import os

# Set page config as the first Streamlit command
st.set_page_config(page_title="Shazam Clone", layout="wide", page_icon="🎵")

# Initialize session state
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = ""

# Load AssemblyAI API key from Streamlit secrets
try:
    aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]
except KeyError:
    st.error("❌ AssemblyAI API key not found. Please set it in Streamlit Cloud secrets as 'ASSEMBLYAI_API_KEY'.")
    st.stop()

# ChromaDB Setup with error handling
try:
    client = chromadb.EphemeralClient()  # Using EphemeralClient for in-memory storage
    collection = client.get_or_create_collection(name="subtitle_chunks")
except Exception as e:
    st.error(f"Failed to initialize ChromaDB: {str(e)}. Using in-memory fallback.")
    client = chromadb.Client()  # Fallback to basic in-memory client
    collection = client.get_or_create_collection(name="subtitle_chunks")

# Populate with sample data if empty (for demo purposes)
@st.cache_resource
def initialize_sample_data():
    if collection.count() == 0:
        sample_subtitles = [
            "I am Groot. We are Groot.",
            "May the Force be with you.",
            "Houston, we have a problem.",
            "My precious...",
            "Life is like a box of chocolates."
        ]
        sample_metadata = [
            {"subtitle_name": "Guardians of the Galaxy", "subtitle_id": "12345"},
            {"subtitle_name": "Star Wars", "subtitle_id": "23456"},
            {"subtitle_name": "Apollo 13", "subtitle_id": "34567"},
            {"subtitle_name": "The Lord of the Rings", "subtitle_id": "45678"},
            {"subtitle_name": "Forrest Gump", "subtitle_id": "56789"}
        ]
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sample_subtitles, show_progress_bar=False).tolist()
        ids = [f"chunk_{i}" for i in range(len(sample_subtitles))]
        collection.add(
            documents=sample_subtitles,
            metadatas=sample_metadata,
            embeddings=embeddings,
            ids=ids
        )
    return collection

# Initialize sample data
collection = initialize_sample_data()

@backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=60)
def load_embedding_model():
    """Loads the SentenceTransformer model with retry logic."""
    try:
        with st.spinner("Loading embedding model (this may take a moment)..."):
            response = requests.get("https://huggingface.co", timeout=10)
            if response.status_code != 200:
                raise Exception("Cannot reach Hugging Face servers.")
            return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(
            f"Failed to load embedding model: {str(e)}. "
            "Please check your internet connection, wait a moment, and reboot the app."
        )
        st.stop()

def transcribe_audio(audio_file):
    """Transcribes audio using AssemblyAI."""
    if audio_file is None:
        return "Please upload an audio file."
    
    temp_file = "temp_audio_file"
    try:
        with open(temp_file, "wb") as f:
            f.write(audio_file.getbuffer())
        
        config = aai.TranscriptionConfig(language_code="en")
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(temp_file)
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return transcript.text if transcript.text else "No transcription available"
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return f"Error: {str(e)}"

def retrieve_and_display_results(query, top_n):
    """Retrieves top N subtitle search results based on query."""
    if not query or "Error" in query:
        return json.dumps([{"Result": "No valid transcription available."}], indent=4)

    model = load_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()

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
            subtitle_text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            
            subtitle_name = metadata.get("subtitle_name", "Unknown")
            subtitle_id = metadata.get("subtitle_id", "N/A")
            url = f"https://www.opensubtitles.org/en/subtitles/{subtitle_id}"

            formatted_results.append({
                "Result": i + 1,
                "Subtitle Name": subtitle_name.upper(),
                "Subtitle Text": subtitle_text,
                "URL": url,
            })

        return json.dumps(formatted_results, indent=4)
    
    return json.dumps([{"Result": "No results found"}], indent=4)

def clear_all():
    """Clears the transcribed text and search results."""
    st.session_state.transcribed_text = ""
    st.session_state.search_results = ""

def main():
    st.title("🎵 Shazam Clone: Audio Transcription & Subtitle Search")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        top_n_results = st.slider("Select Number of Results:", min_value=1, max_value=10, value=5)
        audio_input = st.file_uploader("📂 Upload Audio File", type=["wav", "mp3"])
        
        if st.button("🚀 Transcribe & Search"):
            if audio_input:
                with st.spinner("Processing audio..."):
                    transcribed_text = transcribe_audio(audio_input)
                    st.session_state.transcribed_text = transcribed_text
                    st.session_state.search_results = retrieve_and_display_results(transcribed_text, top_n_results)
            else:
                st.warning("⚠️ Please upload an audio file first.")
    
    st.subheader("📝 Transcribed Text")
    st.text_area("", value=st.session_state.transcribed_text, height=150)
    
    st.subheader("🔍 Subtitle Search Results")
    st.json(st.session_state.search_results)
    
    if st.button("🧹 Clear All"):
        clear_all()
        st.rerun()

if __name__ == "__main__":
    main()
