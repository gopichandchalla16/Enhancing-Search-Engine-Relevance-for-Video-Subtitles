import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Shazam Clone", layout="wide", page_icon="🎵")

import numpy as np
import json
import os
from pathlib import Path
import time
import pandas as pd
from datetime import datetime

# Check for required packages and install if missing
import sys
import subprocess

# Function to check and install missing packages
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        st.warning(f"Installing required package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        st.experimental_rerun()

# Install required packages
install_package("assemblyai")
install_package("sentence_transformers")

# Now try imports with better error handling
try:
    import assemblyai as aai
    from sentence_transformers import SentenceTransformer
    try:
        import plotly.express as px
    except ImportError:
        install_package("plotly")
        import plotly.express as px
except Exception as e:
    st.error(f"Error importing required libraries: {str(e)}")
    st.info("Please make sure all required libraries are installed by running: pip install streamlit numpy assemblyai sentence-transformers plotly pandas")
    st.stop()

# Initialize session state variables if they don't exist
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

# Securely handle AssemblyAI API key
def get_api_key():
    # For deployment on Streamlit Cloud, use secrets management
    if "general" in st.secrets and "ASSEMBLYAI_API_KEY" in st.secrets["general"]:
        return st.secrets["general"]["ASSEMBLYAI_API_KEY"]
    else:
        # Fallback to environment variable if available
        api_key = os.environ.get("ASSEMBLYAI_API_KEY")
        if api_key:
            return api_key
        else:
            # If running locally and .streamlit/secrets.toml doesn't exist
            # Prompt for API key in the app
            if "api_key" not in st.session_state:
                st.session_state.api_key = ""
            
            api_key = st.text_input(
                "Enter your AssemblyAI API key:",
                value=st.session_state.api_key,
                type="password"
            )
            st.session_state.api_key = api_key
            
            if not api_key:
                st.warning("⚠ Please enter your AssemblyAI API key to continue.")
                st.info("You can get an API key from https://www.assemblyai.com/")
                st.stop()
            return api_key

# Set the API key safely
try:
    aai.settings.api_key = get_api_key()
except Exception as e:
    st.error(f"Error setting AssemblyAI API key: {str(e)}")
    st.info("Please check your API key and try again.")
    st.stop()

# Create a simple in-memory database instead of using ChromaDB
class SimpleVectorDB:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.model = None
        
    def initialize_model(self):
        if self.model is None:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                st.error(f"Error loading SentenceTransformer model: {str(e)}")
                st.info("Try reloading the page or check your internet connection.")
                st.stop()
        
    def add(self, documents, metadatas=None, embeddings=None):
        self.initialize_model()
        
        if embeddings is None:
            embeddings = self.model.encode(documents, show_progress_bar=False)
        
        for i, doc in enumerate(documents):
            self.documents.append(doc)
            self.embeddings.append(embeddings[i])
            self.metadata.append(metadatas[i] if metadatas else {})
            
    def query(self, query_embeddings, n_results=5, include=None):
        if not self.embeddings:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
        query_embedding = query_embeddings[0]
        
        # Compute cosine similarity
        similarities = []
        for emb in self.embeddings:
            # Improved similarity calculation with normalization
            dot_product = sum(a*b for a, b in zip(query_embedding, emb))
            magnitude1 = sum(a*a for a in query_embedding) ** 0.5
            magnitude2 = sum(b*b for b in emb) ** 0.5
            sim = dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0
            similarities.append(sim)
            
        # Get top n results
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:n_results]
        
        result = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        for idx in top_indices:
            result["documents"][0].append(self.documents[idx])
            result["metadatas"][0].append(self.metadata[idx])
            result["distances"][0].append(similarities[idx])
            
        return result

# Initialize our simple vector database
@st.cache_resource
def get_vector_db():
    return SimpleVectorDB()

# Get or create the database
db = get_vector_db()

# Sample data for demo purposes - in a real app, you'd load this from files
sample_subtitles = [
    "I am Groot. I am Groot. I am Groot.",
    "May the Force be with you.",
    "Houston, we have a problem.",
    "My precious... my precious...",
    "Life is like a box of chocolates.",
    "I'll be back.",
    "Bond. James Bond.",
    "There's no place like home.",
    "E.T. phone home.",
    "To infinity and beyond!"
]

sample_metadata = [
    {"subtitle_name": "Guardians of the Galaxy", "subtitle_id": "12345", "year": "2014", "genre": "Sci-Fi"},
    {"subtitle_name": "Star Wars", "subtitle_id": "23456", "year": "1977", "genre": "Sci-Fi"},
    {"subtitle_name": "Apollo 13", "subtitle_id": "34567", "year": "1995", "genre": "Drama"},
    {"subtitle_name": "The Lord of the Rings", "subtitle_id": "45678", "year": "2001", "genre": "Fantasy"},
    {"subtitle_name": "Forrest Gump", "subtitle_id": "56789", "year": "1994", "genre": "Drama"},
    {"subtitle_name": "The Terminator", "subtitle_id": "67890", "year": "1984", "genre": "Action"},
    {"subtitle_name": "James Bond Series", "subtitle_id": "78901", "year": "1962", "genre": "Action"},
    {"subtitle_name": "The Wizard of Oz", "subtitle_id": "89012", "year": "1939", "genre": "Fantasy"},
    {"subtitle_name": "E.T.", "subtitle_id": "90123", "year": "1982", "genre": "Sci-Fi"},
    {"subtitle_name": "Toy Story", "subtitle_id": "01234", "year": "1995", "genre": "Animation"}
]

# Add sample data if the database is empty
if not db.documents:
    try:
        db.initialize_model()
        db.add(sample_subtitles, sample_metadata)
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        # Continue without sample data

def transcribe_audio(audio_file):
    """Transcribes audio using AssemblyAI with enhanced error handling and profiling."""
    if audio_file is None:
        return "Please upload an audio file.", None, None, None
    
    start_time = time.time()
    
    try:
        # Create a unique temp filename to avoid conflicts
        timestamp = int(time.time())
        temp_file_path = f"temp_audio_{timestamp}_{audio_file.name}"
        
        with open(temp_file_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        # Configure and transcribe with more options
        config = aai.TranscriptionConfig(
            language_code="en",
            punctuate=True,
            format_text=True,
            speaker_labels=True
        )
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(temp_file_path)
        
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Calculate confidence score (average word-level confidence)
        confidence = None
        try:
            if hasattr(transcript, 'words') and transcript.words:
                confidences = [word.confidence for word in transcript.words if hasattr(word, 'confidence')]
                if confidences:
                    confidence = sum(confidences) / len(confidences)
        except Exception:
            pass
            
        return transcript.text, transcript.text, confidence, processing_time
    except Exception as e:
        error_msg = f"Transcription error: {str(e)}"
        st.error(error_msg)
        
        # Clean up temp file if it exists
        try:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception:
            pass
            
        return error_msg, None, None, time.time() - start_time

def retrieve_and_display_results(query, top_n):
    """Retrieves top N subtitle search results based on query with enhanced metrics."""
    if not query:
        return json.dumps([{"Result": "No transcription text available for search."}], indent=4)

    try:
        db.initialize_model()
        query_embedding = db.model.encode([query], show_progress_bar=False).tolist()

        results = db.query(
            query_embeddings=query_embedding,
            n_results=min(top_n, 10)  # Limit to available results
        )

        return format_results_as_json(results)
    except Exception as e:
        error_msg = f"Search error: {e}"
        st.error(error_msg)
        return json.dumps([{"Result": error_msg}], indent=4)

def format_results_as_json(results):
    """Formats retrieved subtitle search results with enhanced information."""
    formatted_results = []

    if results and "documents" in results and results["documents"] and len(results["documents"][0]) > 0:
        for i in range(len(results["documents"][0])):
            subtitle_text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            subtitle_name = metadata.get("subtitle_name", "Unknown")
            subtitle_id = metadata.get("subtitle_id", "N/A")
            year = metadata.get("year", "Unknown")
            genre = metadata.get("genre", "Unknown")
            
            # Get similarity score if available
            similarity_score = None
            if "distances" in results and results["distances"] and len(results["distances"][0]) > i:
                similarity_score = results["distances"][0][i]
                
            url = f"https://www.opensubtitles.org/en/subtitles/{subtitle_id}"

            formatted_results.append({
                "Result": i + 1,
                "Similarity": f"{similarity_score:.4f}" if similarity_score is not None else "N/A",
                "Media Title": subtitle_name.upper(),
                "Year": year,
                "Genre": genre,
                "Quote": subtitle_text,
                "URL": url,
            })

        return json.dumps(formatted_results, indent=4)
    
    return json.dumps([{"Result": "No matching subtitles found"}], indent=4)

def clear_all():
    """Clears the transcribed text and search results."""
    st.session_state.transcribed_text = ""
    st.session_state.search_results = "{}"
    st.session_state.confidence_score = None
    st.session_state.processing_time = None

def save_search_history(query, results_json):
    """Saves search query and timestamp to history."""
    try:
        results = json.loads(results_json)
        first_result = results[0] if results and len(results) > 0 else {"Media Title": "No results"}
        
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query[:50] + "..." if len(query) > 50 else query,
            "top_result": first_result.get("Media Title", "Unknown") if "Result" not in first_result or first_result["Result"] != "No matching subtitles found" else "No matches",
            "num_results": len(results) if "Result" not in results[0] or results[0]["Result"] != "No matching subtitles found" else 0
        }
        
        st.session_state.search_history.append(history_entry)
        
        # Keep only the most recent 20 searches
        if len(st.session_state.search_history) > 20:
            st.session_state.search_history = st.session_state.search_history[-20:]
    except Exception:
        # Silently handle any issues with history saving
        pass

def display_metrics():
    """Displays performance metrics in a dashboard-like layout."""
    if st.session_state.processing_time or st.session_state.confidence_score:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.processing_time:
                st.metric("Processing Time", f"{st.session_state.processing_time:.2f}s")
                
        with col2:
            if st.session_state.confidence_score:
                confidence_pct = st.session_state.confidence_score * 100
                st.metric("Transcription Confidence", f"{confidence_pct:.1f}%")

def display_search_history():
    """Displays search history in a tabular format with visualization."""
    if not st.session_state.search_history:
        st.info("No search history available yet.")
        return
        
    st.subheader("📊 Search History")
    
    # Create a DataFrame for the history table
    history_df = pd.DataFrame(st.session_state.search_history)
    
    # Display the table
    st.dataframe(history_df, use_container_width=True)
    
    # If we have enough data, show a visualization
    if len(history_df) >= 3:
        try:
            # Count results by top match
            result_counts = history_df['top_result'].value_counts().reset_index()
            result_counts.columns = ['Media', 'Count']
            
            # Create a simple bar chart
            fig = px.bar(
                result_counts, 
                x='Media', 
                y='Count',
                title='Most Frequent Matches',
                labels={'Media': 'Media Title', 'Count': 'Number of Matches'},
                color='Count',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            # Silently handle visualization errors
            pass

def display_theme_customization():
    """Provides theme customization options."""
    st.sidebar.subheader("🎨 Theme Customization")
    
    themes = {
        "Default": {"primary": "#F63366", "background": "#FFFFFF", "text": "#262730"},
        "Dark Mode": {"primary": "#00CCFF", "background": "#262730", "text": "#FAFAFA"},
        "Nature": {"primary": "#4CAF50", "background": "#F1F8E9", "text": "#1B5E20"},
        "Ocean": {"primary": "#2196F3", "background": "#E3F2FD", "text": "#0D47A1"}
    }
    
    selected_theme = st.sidebar.selectbox("Select Theme", list(themes.keys()), index=0)
    theme = themes[selected_theme]
    
    # Apply theme using custom CSS
    css = f"""
    <style>
        .stApp {{
            background-color: {theme['background']};
            color: {theme['text']};
        }}
        .stButton>button {{
            background-color: {theme['primary']};
            color: white;
        }}
        .stProgress .st-bo {{
            background-color: {theme['primary']};
        }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

def main():
    st.title("🎵 Advanced Audio Recognition System")
    st.markdown("### Identify and search for media quotes from audio clips")
    
    # Apply theme customization
    display_theme_customization()
    
    # Create tabbed interface
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 History", "ℹ About"])
    
    with tab1:
        with st.sidebar:
            st.header("⚙ Settings")
            
            # Advanced settings collapsible section
            with st.expander("Advanced Settings", expanded=False):
                top_n_results = st.slider("Results to Display:", min_value=1, max_value=10, value=5)
                min_confidence = st.slider("Minimum Confidence Threshold:", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
                lang_options = ["English", "Spanish", "French", "German", "Japanese"]
                selected_lang = st.selectbox("Audio Language:", lang_options, index=0)
                
                # Map selected language to language code
                lang_codes = {"English": "en", "Spanish": "es", "French": "fr", "German": "de", "Japanese": "ja"}
                lang_code = lang_codes.get(selected_lang, "en")
            
            # Main audio upload and processing
            st.subheader("📂 Upload Audio")
            audio_input = st.file_uploader("Upload a voice or audio clip", type=["wav", "mp3", "m4a", "ogg"])
            
            if audio_input:
                st.audio(audio_input, format=f"audio/{audio_input.name.split('.')[-1]}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Analyze", use_container_width=True):
                    with st.spinner("Processing audio..."):
                        if audio_input:
                            # Show progress indication
                            progress_bar = st.progress(0)
                            
                            # Update progress to show activity
                            for i in range(5):
                                progress_bar.progress((i+1)*10)
                                time.sleep(0.1)
                                
                            transcribed_text, raw_text, confidence, proc_time = transcribe_audio(audio_input)
                            st.session_state.transcribed_text = transcribed_text
                            st.session_state.confidence_score = confidence
                            st.session_state.processing_time = proc_time
                            
                            progress_bar.progress(70)
                            
                            if transcribed_text and not transcribed_text.startswith("Error"):
                                result_json = retrieve_and_display_results(transcribed_text, top_n_results)
                                st.session_state.search_results = result_json
                                
                                # Save to search history
                                save_search_history(transcribed_text, result_json)
                            else:
                                st.session_state.search_results = json.dumps([{"Result": "Transcription failed"}], indent=4)
                                
                            progress_bar.progress(100)
                            time.sleep(0.5)
                            progress_bar.empty()
                        else:
                            st.warning("⚠ Please upload an audio file first.")
            
            with col2:
                if st.button("🧹 Clear", use_container_width=True):
                    clear_all()
                    st.experimental_rerun()
        
        # Display performance metrics
        display_metrics()
        
        # Display transcribed text and search results
        st.subheader("📝 Transcribed Text")
        st.text_area("", value=st.session_state.get("transcribed_text", ""), height=100, key="text_display")
        
        st.subheader("🔍 Search Results")
        try:
            results = json.loads(st.session_state.get("search_results", "{}"))
            
            # Display results as cards instead of raw JSON
            if results and isinstance(results, list) and len(results) > 0:
                if "Result" in results[0] and results[0]["Result"] == "No matching subtitles found":
                    st.info("No matching quotes found in our database.")
                elif "Result" not in results[0] or isinstance(results[0]["Result"], int):
                    # Create columns for the cards
                    num_cols = 2
                    for i in range(0, len(results), num_cols):
                        cols = st.columns(num_cols)
                        for j in range(num_cols):
                            if i+j < len(results):
                                result = results[i+j]
                                with cols[j]:
                                    st.markdown(f"""
                                    <div style="border:1px solid #ccc; border-radius:5px; padding:10px; margin-bottom:10px;">
                                        <h4>{result.get('Media Title', 'Unknown')}</h4>
                                        <p><em>"{result.get('Quote', '')}"</em></p>
                                        <p>Year: {result.get('Year', 'Unknown')} | Genre: {result.get('Genre', 'Unknown')}</p>
                                        <p>Match Score: {result.get('Similarity', 'N/A')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                else:
                    st.warning(results[0].get("Result", "Unknown error"))
            else:
                st.info("Upload and analyze audio to see results.")
        except Exception as e:
            st.warning(f"Error displaying results: {str(e)}")
    
    with tab2:
        display_search_history()
    
    with tab3:
        st.subheader("About This Application")
        st.markdown("""
        ### Advanced Audio Recognition System
        
        This application allows you to identify movie quotes, famous lines, and other media content from audio clips. 
        
        #### How It Works:
        1. Upload an audio clip containing speech
        2. Our system transcribes the speech into text
        3. The transcribed text is compared against a database of movie quotes and subtitles
        4. Matching results are displayed with details about the source media
        
        #### Features:
        - Real-time audio transcription using AssemblyAI
        - Semantic search for finding approximate matches, not just exact quotes
        - Confidence scoring and processing metrics
        - Search history tracking and visualization
        - Customizable theme and appearance
        
        #### Limitations:
        - The demo version uses a small sample database
        - Best results require clear audio with minimal background noise
        - Currently optimized for English language content
        
        #### Future Enhancements:
        - Expanded media database with thousands of movies and TV shows
        - Music recognition capabilities
        - Real-time streaming audio analysis
        - Multi-language support
        """)

if __name__ == "__main__":
    main()
