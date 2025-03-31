import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
import plotly.express as px
import assemblyai as aai
from sentence_transformers import SentenceTransformer
import requests
import backoff

# Set page config as the first Streamlit command
st.set_page_config(page_title="Video Subtitle Search Engine", layout="wide", page_icon="🎬")

# Initialize session state variables
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = "{}"
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "confidence_score" not in st.session_state:
    st.session_state.confidence_score = None
if "processing_time" not in st.session_state:
    st.session_state.processing_time = None
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

# Set AssemblyAI API key from secrets (for Streamlit Cloud)
try:
    aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]
except KeyError:
    st.error("AssemblyAI API key not found. Please set it in Streamlit Cloud secrets.")
    st.stop()

# Simple in-memory vector database
class SimpleVectorDB:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.model = None
        
    @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=60)
    def initialize_model(self):
        if self.model is None:
            try:
                with st.spinner("Loading embedding model (this may take a moment)..."):
                    # Check internet connectivity
                    response = requests.get("https://huggingface.co", timeout=10)
                    if response.status_code != 200:
                        raise Exception("Cannot reach Hugging Face servers.")
                    
                    self.model = SentenceTransformer("all-MiniLM-L6-v2")
                st.session_state.db_initialized = True
            except Exception as e:
                st.error(
                    f"Failed to load embedding model: {str(e)}. "
                    "This may be due to a network issue. Please check your internet connection, "
                    "wait a moment, and try rebooting the app. If the issue persists, contact support."
                )
                st.stop()
        
    def add_chunks(self, document, metadata, chunk_size=10, overlap=3):
        words = document.split()
        chunks = []
        chunk_metadatas = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if not chunk:
                continue
            chunks.append(chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = f"{metadata.get('subtitle_id', 'unknown')}_{i//chunk_size}"
            chunk_metadatas.append(chunk_metadata)
        
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        self.documents.extend(chunks)
        self.embeddings.extend(embeddings)
        self.metadata.extend(chunk_metadatas)
            
    def query(self, query_embeddings, n_results=5):
        if not self.embeddings:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        query_embedding = query_embeddings[0]
        similarities = [np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb)) 
                        for emb in self.embeddings]
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        result = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        for idx in top_indices:
            result["documents"][0].append(self.documents[idx])
            result["metadatas"][0].append(self.metadata[idx])
            result["distances"][0].append(similarities[idx])
        return result

# Cache the database
@st.cache_resource
def get_vector_db():
    db = SimpleVectorDB()
    db.initialize_model()
    sample_subtitles = [
        "I am Groot. I am Groot. We are Groot.",
        "May the Force be with you. I am your father.",
        "Houston, we have a problem. One small step for man.",
        "My precious... They stole it from us!",
        "Life is like a box of chocolates, you never know what you're gonna get."
    ]
    sample_metadata = [
        {"subtitle_name": "Guardians of the Galaxy", "subtitle_id": "12345"},
        {"subtitle_name": "Star Wars", "subtitle_id": "23456"},
        {"subtitle_name": "Apollo 13", "subtitle_id": "34567"},
        {"subtitle_name": "The Lord of the Rings", "subtitle_id": "45678"},
        {"subtitle_name": "Forrest Gump", "subtitle_id": "56789"}
    ]
    for subtitle, metadata in zip(sample_subtitles, sample_metadata):
        db.add_chunks(subtitle, metadata)
    return db

# Initialize the database
try:
    db = get_vector_db()
except Exception as e:
    st.error(f"Failed to initialize database: {str(e)}. App cannot proceed.")
    st.stop()

def transcribe_audio(audio_file):
    start_time = time.time()
    try:
        temp_file = f"temp_{audio_file.name}"
        with open(temp_file, "wb") as f:
            f.write(audio_file.getbuffer())
        
        config = aai.TranscriptionConfig(language_code="en")
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_file)
        
        os.remove(temp_file)
        processing_time = time.time() - start_time
        
        confidence = None
        if transcript.words:
            confidences = [word.confidence for word in transcript.words]
            confidence = sum(confidences) / len(confidences) if confidences else None
        
        return transcript.text, confidence, processing_time
    except Exception as e:
        os.remove(temp_file) if os.path.exists(temp_file) else None
        return f"Error: {str(e)}", None, time.time() - start_time

def retrieve_results(query, top_n):
    if not query or "Error" in query:
        return json.dumps([{"Result": "No valid transcription available."}], indent=4)
    
    query_embedding = db.model.encode([query], show_progress_bar=False).tolist()
    results = db.query(query_embeddings=query_embedding, n_results=top_n)
    
    formatted_results = []
    for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
        formatted_results.append({
            "Result": i + 1,
            "Similarity": f"{dist:.4f}",
            "Media Title": meta.get("subtitle_name", "Unknown"),
            "Quote": doc,
            "URL": f"https://www.opensubtitles.org/en/subtitles/{meta.get('subtitle_id', 'unknown')}"
        })
    return json.dumps(formatted_results if formatted_results else [{"Result": "No matches found"}], indent=4)

def save_search_history(query, results_json):
    results = json.loads(results_json)
    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query[:50] + "..." if len(query) > 50 else query,
        "top_result": results[0].get("Media Title", "No matches") if results else "No matches",
        "num_results": len(results) if results and "Result" in results[0] and isinstance(results[0]["Result"], int) else 0
    }
    st.session_state.search_history.append(history_entry)
    if len(st.session_state.search_history) > 20:
        st.session_state.search_history = st.session_state.search_history[-20:]

def display_metrics():
    if st.session_state.processing_time or st.session_state.confidence_score:
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.processing_time:
                st.metric("Processing Time", f"{st.session_state.processing_time:.2f}s")
        with col2:
            if st.session_state.confidence_score:
                st.metric("Confidence", f"{st.session_state.confidence_score:.1%}")

def display_history():
    if st.session_state.search_history:
        df = pd.DataFrame(st.session_state.search_history)
        st.subheader("Search History")
        st.dataframe(df)
        if len(df) >= 3:
            fig = px.bar(df["top_result"].value_counts().reset_index(), x="index", y="top_result", 
                         labels={"index": "Media", "top_result": "Count"}, title="Most Frequent Matches")
            st.plotly_chart(fig)

def main():
    st.title("🎬 Video Subtitle Search Engine")
    st.markdown("Find movie quotes from audio clips!")
    
    tab1, tab2 = st.tabs(["Search", "History"])
    
    with tab1:
        audio_input = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])
        if audio_input:
            st.audio(audio_input)
        
        if st.button("Analyze"):
            with st.spinner("Processing..."):
                text, conf, time_taken = transcribe_audio(audio_input)
                st.session_state.transcribed_text = text
                st.session_state.confidence_score = conf
                st.session_state.processing_time = time_taken
                
                results = retrieve_results(text, 5)
                st.session_state.search_results = results
                save_search_history(text, results)
        
        st.subheader("Transcribed Text")
        st.text_area("", st.session_state.transcribed_text, height=100)
        
        st.subheader("Search Results")
        results = json.loads(st.session_state.search_results)
        for result in results:
            if "Result" in result and isinstance(result["Result"], int):
                st.markdown(f"**{result['Media Title']}** (Similarity: {result['Similarity']})")
                st.write(f"Quote: {result['Quote']}")
                st.write(f"[Link]({result['URL']})")
            else:
                st.write(result.get("Result", "No results"))
        
        display_metrics()
    
    with tab2:
        display_history()

if __name__ == "__main__":
    main()
