import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
import plotly.express as px

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Video Subtitle Search Engine", layout="wide", page_icon="🎬")

# Create a requirements.txt file first instead of installing packages on the fly
if not os.path.exists("requirements.txt"):
    with open("requirements.txt", "w") as f:
        f.write("streamlit>=1.24.0\n")
        f.write("numpy>=1.22.0\n")
        f.write("pandas>=1.5.0\n")
        f.write("plotly>=5.13.0\n")
        f.write("sentence-transformers>=2.2.2\n")
        f.write("assemblyai>=0.10.0\n")
    
    st.warning("Created requirements.txt file. Please restart the app or run: pip install -r requirements.txt")
    st.stop()

# Import required packages with better error handling
try:
    import assemblyai as aai
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("Required packages are missing. Please install them by running: pip install -r requirements.txt")
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
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

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

# Create a simple in-memory database for vector embeddings
class SimpleVectorDB:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.model = None
        
    def initialize_model(self):
        if self.model is None:
            try:
                with st.spinner("Loading embedding model..."):
                    # Using all-MiniLM-L6-v2 as it's faster and has good performance
                    self.model = SentenceTransformer("all-MiniLM-L6-v2")
                st.session_state.db_initialized = True
            except Exception as e:
                st.error(f"Error loading SentenceTransformer model: {str(e)}")
                st.info("Try reloading the page or check your internet connection.")
                st.stop()
        
    def add(self, documents, metadatas=None, embeddings=None):
        """Add documents and their embeddings to the database"""
        self.initialize_model()
        
        if embeddings is None:
            with st.spinner("Generating embeddings..."):
                embeddings = self.model.encode(documents, show_progress_bar=False)
        
        for i, doc in enumerate(documents):
            self.documents.append(doc)
            self.embeddings.append(embeddings[i])
            self.metadata.append(metadatas[i] if metadatas else {})
    
    def add_chunks(self, document, metadata=None, chunk_size=128, overlap=20):
        """Chunk a document and add the chunks to the database"""
        # Split the document into words
        words = document.split()
        chunks = []
        chunk_metadatas = []
        
        # Create chunks with overlap
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if not chunk:  # Skip empty chunks
                continue
                
            chunks.append(chunk)
            
            # Create metadata for the chunk with position information
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_id"] = f"{metadata.get('subtitle_id', 'unknown')}_{i//chunk_size}"
            chunk_metadata["start_pos"] = i
            chunk_metadata["end_pos"] = min(i + chunk_size, len(words))
            chunk_metadatas.append(chunk_metadata)
        
        # Add the chunks to the database
        self.add(chunks, chunk_metadatas)
            
    def query(self, query_embeddings, n_results=5, include=None):
        """Find the most similar documents to the query"""
        if not self.embeddings:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
        query_embedding = query_embeddings[0]
        
        # Compute cosine similarity
        similarities = []
        for emb in self.embeddings:
            # Cosine similarity calculation
            dot_product = np.dot(query_embedding, emb)
            magnitude1 = np.linalg.norm(query_embedding)
            magnitude2 = np.linalg.norm(emb)
            sim = dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0
            similarities.append(sim)
            
        # Get top n results
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        result = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        for idx in top_indices:
            result["documents"][0].append(self.documents[idx])
            result["metadatas"][0].append(self.metadata[idx])
            result["distances"][0].append(similarities[idx])
            
        return result

# Initialize our vector database
@st.cache_resource
def get_vector_db():
    return SimpleVectorDB()

# Get or create the database
db = get_vector_db()

# Load sample subtitle data
def load_sample_data():
    """Load sample subtitle data for demonstration purposes"""
    # Extended sample data with longer quotes
    sample_subtitles = [
        "I am Groot. I am Groot. I am Groot. We are Groot. I am Groot. I am Groot. We are Groot.",
        "May the Force be with you. The Force will be with you, always. I am your father. Do or do not, there is no try.",
        "Houston, we have a problem. Failure is not an option. One small step for man, one giant leap for mankind.",
        "My precious... my precious... Filthy little hobbitses. They stole it from us! We wants it back! The precious!",
        "Life is like a box of chocolates, you never know what you're gonna get. Run, Forrest, run! Stupid is as stupid does.",
        "I'll be back. Hasta la vista, baby. Come with me if you want to live. It's in your nature to destroy yourselves.",
        "Bond. James Bond. Shaken, not stirred. The name's Bond, James Bond. I never miss.",
        "There's no place like home. Follow the yellow brick road. I'll get you, my pretty, and your little dog, too!",
        "E.T. phone home. E.T. home phone. Be good. I'll be right here.",
        "To infinity and beyond! You've got a friend in me. This isn't flying, this is falling with style!"
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
    
    # Process each subtitle document
    for i, subtitle in enumerate(sample_subtitles):
        # Add document chunks instead of full documents for better search results
        db.add_chunks(subtitle, sample_metadata[i], chunk_size=10, overlap=3)

# Add sample data if the database is empty
if not db.documents:
    try:
        db.initialize_model()
        load_sample_data()
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        # Continue without sample data

def transcribe_audio(audio_file):
    """Transcribes audio using AssemblyAI with enhanced error handling and profiling"""
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
    """Retrieves top N subtitle search results based on query with enhanced metrics"""
    if not query:
        return json.dumps([{"Result": "No transcription text available for search."}], indent=4)

    try:
        # Make sure the database model is initialized
        db.initialize_model()
        
        # Generate embedding for the query
        query_embedding = db.model.encode([query], show_progress_bar=False).tolist()

        # Search for similar subtitles
        results = db.query(
            query_embeddings=query_embedding,
            n_results=min(top_n, len(db.documents))  # Limit to available results
        )

        return format_results_as_json(results)
    except Exception as e:
        error_msg = f"Search error: {e}"
        st.error(error_msg)
        return json.dumps([{"Result": error_msg}], indent=4)

def format_results_as_json(results):
    """Formats retrieved subtitle search results with enhanced information"""
    formatted_results = []

    if results and "documents" in results and results["documents"] and len(results["documents"][0]) > 0:
        # Group results by subtitle_id to deduplicate and merge chunks
        grouped_results = {}
        
        for i in range(len(results["documents"][0])):
            subtitle_text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            similarity_score = results["distances"][0][i] if "distances" in results else 0
            
            subtitle_id = metadata.get("subtitle_id", "unknown")
            
            # Initialize group or update if score is higher
            if subtitle_id not in grouped_results or similarity_score > grouped_results[subtitle_id]["score"]:
                grouped_results[subtitle_id] = {
                    "subtitle_name": metadata.get("subtitle_name", "Unknown"),
                    "subtitle_id": subtitle_id,
                    "year": metadata.get("year", "Unknown"),
                    "genre": metadata.get("genre", "Unknown"),
                    "score": similarity_score,
                    "text": subtitle_text
                }
        
        # Format the grouped results
        for i, (_, result) in enumerate(sorted(grouped_results.items(), key=lambda x: x[1]["score"], reverse=True)):
            url = f"https://www.opensubtitles.org/en/subtitles/{result['subtitle_id']}"
            
            formatted_results.append({
                "Result": i + 1,
                "Similarity": f"{result['score']:.4f}",
                "Media Title": result["subtitle_name"].upper(),
                "Year": result["year"],
                "Genre": result["genre"],
                "Quote": result["text"],
                "URL": url,
            })

        return json.dumps(formatted_results, indent=4)
    
    return json.dumps([{"Result": "No matching subtitles found"}], indent=4)

def clear_all():
    """Clears the transcribed text and search results"""
    st.session_state.transcribed_text = ""
    st.session_state.search_results = "{}"
    st.session_state.confidence_score = None
    st.session_state.processing_time = None

def save_search_history(query, results_json):
    """Saves search query and timestamp to history"""
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
    """Displays performance metrics in a dashboard-like layout"""
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
    """Displays search history in a tabular format with visualization"""
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
    """Provides theme customization options"""
    st.sidebar.subheader("🎨 Theme Customization")
    
    themes = {
        "Default": {"primary": "#F63366", "background": "#FFFFFF", "text": "#262730"},
        "Dark Mode": {"primary": "#00CCFF", "background": "#262730", "text": "#FAFAFA"},
        "Cinema": {"primary": "#FF5722", "background": "#121212", "text": "#FAFAFA"},
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
    st.title("🎬 Video Subtitle Search Engine")
    st.markdown("### Find movie and TV quotes from audio clips")
    
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
                chunk_size = st.slider("Chunk Size:", min_value=5, max_value=50, value=10)
                overlap = st.slider("Chunk Overlap:", min_value=0, max_value=10, value=3)
                
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
                                
                            # Transcribe the audio
                            transcribed_text, raw_text, confidence, proc_time = transcribe_audio(audio_input)
                            st.session_state.transcribed_text = transcribed_text
                            st.session_state.confidence_score = confidence
                            st.session_state.processing_time = proc_time
                            
                            progress_bar.progress(70)
                            
                            # If transcription succeeded, perform search
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
        ### Video Subtitle Search Engine
        
        This application allows you to identify movie quotes, famous lines, and other media content from audio clips. It's like Shazam, but for movie and TV show quotes!
        
        #### How It Works:
        1. **Audio Transcription**: Upload an audio clip containing speech, which is then transcribed using AssemblyAI's API.
        2. **Semantic Search**: The transcribed text is compared against a database of movie quotes and subtitles using sentence embeddings.
        3. **Vector Similarity**: We use cosine similarity between vector embeddings to find the most relevant matches.
        4. **Document Chunking**: Long documents are broken into smaller chunks with overlap to improve search accuracy.
        
        #### Technical Implementation:
        - **Sentence Embeddings**: Using BERT-based models from SentenceTransformers to create semantic representations
        - **Document Chunking**: Breaking subtitles into smaller, overlapping pieces to enhance search precision
        - **Vector Database**: Storing and querying vector embeddings efficiently
        - **Semantic vs. Keyword Search**: Going beyond exact keyword matching to understand context and meaning
        
        #### Limitations:
        - The demo version uses a small sample database
        - Best results require clear audio with minimal background noise
        - Currently optimized for English language content
        
        #### Future Enhancements:
        - Integration with subtitle databases
        - Real-time streaming audio analysis
        - Multi-language support
        - Advanced filtering by genre, year, and more
        """)

if __name__ == "__main__":
    main()
