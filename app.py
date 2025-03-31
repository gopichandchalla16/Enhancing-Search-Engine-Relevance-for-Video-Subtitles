import streamlit as st
import numpy as np
import json
import assemblyai as aai
from sentence_transformers import SentenceTransformer
import chromadb
import os
import time

# Set page config
st.set_page_config(page_title="Shazam Clone", layout="wide", page_icon="🎵")

# Initialize session state
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = ""
if "processing" not in st.session_state:
    st.session_state.processing = False

# Load AssemblyAI API key from Streamlit secrets
try:
    aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]
except KeyError:
    st.error("❌ AssemblyAI API key not found. Please set it in Streamlit secrets as 'ASSEMBLYAI_API_KEY'.")
    st.stop()

# ChromaDB Setup (in-memory for Streamlit Cloud)
client = chromadb.Client(chromadb.config.Settings(is_persistent=False))
collection = client.get_or_create_collection(name="subtitle_chunks")

# Populate sample data if collection is empty
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
initialize_sample_data()

def transcribe_audio(audio_file):
    """Transcribes audio using AssemblyAI."""
    if audio_file is None:
        return "Please upload an audio file.", None
    try:
        with st.spinner("Transcribing audio..."):
            config = aai.TranscriptionConfig(language_code="en")
            transcriber = aai.Transcriber(config=config)
            transcript = transcriber.transcribe(audio_file)
            return transcript.text if transcript.text else "No transcription available.", transcript.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}", None

def retrieve_and_display_results(query, top_n):
    """Retrieves top Lus subtitle search results based on query."""
    if not query or "Error" in query:
        return json.dumps([{"Result": "No valid transcription available."}], indent=4)
    try:
        with st.spinner("Searching subtitles..."):
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_embedding = model.encode([query], show_progress_bar=False).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_n,
                include=["documents", "metadatas"]
            )
            return format_results_as_json(results)
    except Exception as e:
        return json.dumps([{"Result": f"Search error: {str(e)}"}], indent=4)

def format_results_as_json(results):
    """Formats retrieved subtitle search results."""
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
    st.session_state.processing = False

def main():
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .main-title { font-size: 2.5em; color: #1f77b4; }
        .sidebar .sidebar-content { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
        .stButton>button { background-color: #1f77b4; color: white; border-radius: 5px; }
        .stButton>button:hover { background-color: #135e96; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">🎵 Shazam Clone: Audio Transcription & Subtitle Search</h1>', unsafe_allow_html=True)
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])

    with col1:
        with st.sidebar:
            st.header("⚙️ Controls")
            top_n_results = st.slider("Number of Results", min_value=1, max_value=10, value=5, help="Select how many subtitle matches to display.")
            audio_input = st.file_uploader("📂 Upload Audio", type=["wav", "mp3"], help="Upload a WAV or MP3 file to transcribe.")
            
            if st.button("🚀 Transcribe & Search", disabled=st.session_state.processing):
                if audio_input:
                    st.session_state.processing = True
                    transcribed_text, raw_text = transcribe_audio(audio_input)
                    st.session_state.transcribed_text = transcribed_text
                    if raw_text:
                        st.session_state.search_results = retrieve_and_display_results(raw_text, top_n_results)
                    else:
                        st.session_state.search_results = json.dumps([{"Result": "Transcription failed"}], indent=4)
                    st.session_state.processing = False
                    st.rerun()
                else:
                    st.warning("⚠️ Please upload an audio file first.")

            if st.button("🧹 Clear All", disabled=st.session_state.processing):
                clear_all()
                st.rerun()

    with col2:
        st.subheader("📝 Transcribed Text")
        st.text_area("", value=st.session_state.transcribed_text, height=150, disabled=True, help="Transcription of the uploaded audio.")

        st.subheader("🔍 Search Results")
        st.json(st.session_state.search_results, expanded=True)

        # Status indicator
        if st.session_state.processing:
            st.info("Processing... Please wait.")

if __name__ == "__main__":
    main()
