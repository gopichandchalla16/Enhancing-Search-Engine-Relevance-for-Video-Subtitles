import streamlit as st
import numpy as np
import librosa
import os
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import platform
import pandas as pd

st.set_page_config(
    page_title="Audio Fingerprinting System - Shazam Clone",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    .success-box {
        padding: 1rem;
        background-color: #d1fae5;
        border-left: 5px solid #10b981;
        border-radius: 5px;
    }
    .error-box {
        padding: 1rem;
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Function to check if FFmpeg is installed
def check_ffmpeg():
    try:
        # Try running FFmpeg command
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Function to detect FFmpeg path
def detect_ffmpeg_path():
    system = platform.system()
    
    if system == "Windows":
        paths_to_check = [
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            os.path.expanduser("~") + r"\ffmpeg\bin\ffmpeg.exe"
        ]
    elif system == "Darwin":  # macOS
        paths_to_check = [
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
            "/opt/local/bin/ffmpeg"
        ]
    else:  # Linux & others
        paths_to_check = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/ffmpeg/bin/ffmpeg"
        ]
    
    for path in paths_to_check:
        if os.path.isfile(path):
            return path
    
    return None

# Function to extract audio features
def extract_features(audio_path, ffmpeg_path=None):
    try:
        # If ffmpeg_path is provided, set environment variable for librosa
        if ffmpeg_path:
            os.environ["FFMPEG_BINARY"] = ffmpeg_path
            
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract features
        # Mel-frequency cepstral coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # Combine all features
        features = np.hstack([mfcc_mean, spectral_centroid_mean, chroma_mean])
        
        return features
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

# Database of reference tracks
def get_reference_tracks():
    # In a real system, this would load from a database
    # For this demo, we'll use a simple dictionary
    return {
        "Track 1 - Somebody That I Used To Know": "https://example.com/track1.mp3",
        "Track 2 - Despacito": "https://example.com/track2.mp3", 
        "Track 3 - Shape of You": "https://example.com/track3.mp3",
        "Track 4 - Uptown Funk": "https://example.com/track4.mp3",
        "Track 5 - Blinding Lights": "https://example.com/track5.mp3"
    }

# Dummy feature database (in a real app, this would be loaded from a file or database)
def get_reference_features():
    # This simulates the pre-computed features for the reference tracks
    # In a real app, you would load these from a database or compute them in advance
    np.random.seed(42)  # For reproducibility
    reference_features = {}
    
    for track in get_reference_tracks().keys():
        # Generate random features for demonstration
        # Real features would be extracted from actual audio files
        features = np.random.rand(25)  # 13 MFCCs + 1 spectral centroid + 12 chroma features
        reference_features[track] = features
    
    return reference_features

# Function to identify the song
def identify_song(query_features, reference_features):
    best_match = None
    highest_similarity = -1
    
    for track, features in reference_features.items():
        # Calculate cosine similarity
        similarity = cosine_similarity([query_features], [features])[0][0]
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = track
    
    # Set a threshold for identification
    if highest_similarity > 0.7:
        return best_match, highest_similarity
    else:
        return None, highest_similarity

# Main app
def main():
    st.title("🎵 Audio Fingerprinting System - Shazam Clone")
    
    # Check if FFmpeg is installed
    ffmpeg_installed = check_ffmpeg()
    ffmpeg_path = st.session_state.get('ffmpeg_path', detect_ffmpeg_path())
    
    # FFmpeg setup section
    with st.expander("🛠️ FFmpeg Setup", expanded=not ffmpeg_installed):
        if ffmpeg_installed:
            st.markdown('<div class="success-box">✅ FFmpeg is installed and working properly!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">❌ FFmpeg not found! Please install FFmpeg on your system:</div>', unsafe_allow_html=True)
            
            st.markdown("""
            * **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
            * **macOS**: `brew install ffmpeg`
            * **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)
            """)
            
            st.subheader("Specify FFmpeg path manually")
            manual_path = st.text_input("FFmpeg executable path", value=ffmpeg_path if ffmpeg_path else "")
            
            if st.button("Verify Path"):
                if manual_path and os.path.isfile(manual_path):
                    try:
                        subprocess.run([manual_path, "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        st.session_state['ffmpeg_path'] = manual_path
                        st.markdown('<div class="success-box">✅ FFmpeg found at specified path!</div>', unsafe_allow_html=True)
                        ffmpeg_installed = True
                        ffmpeg_path = manual_path
                    except:
                        st.markdown('<div class="error-box">❌ Invalid FFmpeg executable at specified path.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-box">❌ File not found at specified path.</div>', unsafe_allow_html=True)
    
    # Main app content
    st.subheader("Upload an audio sample to identify")
    
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "m4a"])
    
    if uploaded_file and (ffmpeg_installed or ffmpeg_path):
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_path = tmp_file.name
        
        st.audio(uploaded_file, format=f'audio/{os.path.splitext(uploaded_file.name)[1][1:]}')
        
        if st.button("Identify Song"):
            with st.spinner("Processing audio..."):
                # Extract features from the uploaded audio
                query_features = extract_features(audio_path, ffmpeg_path)
                
                if query_features is not None:
                    # Get reference features
                    reference_features = get_reference_features()
                    
                    # Identify the song
                    match, confidence = identify_song(query_features, reference_features)
                    
                    if match:
                        st.success(f"🎵 Identified as: **{match}**")
                        st.progress(float(confidence))
                        st.write(f"Confidence: {confidence:.2f}")
                        
                        # Display additional track information
                        st.subheader("Track Information")
                        
                        # Create a mock data frame with track details
                        data = {
                            "Property": ["Artist", "Album", "Released", "Genre", "Duration"],
                            "Value": ["Unknown Artist", "Unknown Album", "2023", "Pop", "3:45"]
                        }
                        
                        st.table(pd.DataFrame(data))
                    else:
                        st.error("❌ No match found in the database.")
                        st.write(f"Highest similarity score: {confidence:.2f}")
                else:
                    st.error("❌ Failed to extract features from the audio file.")
        
        # Clean up the temporary file
        os.unlink(audio_path)
    elif not ffmpeg_installed and not ffmpeg_path:
        st.warning("⚠️ Please install FFmpeg or provide a valid path to continue.")

    # App information
    st.sidebar.title("About")
    st.sidebar.info("""
    This app demonstrates a simplified version of audio fingerprinting, 
    similar to how Shazam works. It extracts audio features from uploaded files 
    and compares them against a database of reference tracks.
    """)
    
    # Display theme selector in sidebar
    st.sidebar.subheader("UI Theme")
    theme = st.sidebar.selectbox(
        "Select a theme",
        ["Default", "Dark Blue", "Vibrant Purple", "Forest Green"]
    )
    
    # Apply selected theme
    if theme == "Dark Blue":
        st.markdown("""
        <style>
            .main {background-color: #0f172a;}
            h1, h2, h3, p {color: #e2e8f0;}
            .stButton>button {background-color: #3b82f6; color: white;}
            .stButton>button:hover {background-color: #2563eb;}
        </style>
        """, unsafe_allow_html=True)
    elif theme == "Vibrant Purple":
        st.markdown("""
        <style>
            .main {background-color: #faf5ff;}
            h1, h2, h3 {color: #7e22ce;}
            p {color: #581c87;}
            .stButton>button {background-color: #a855f7; color: white;}
            .stButton>button:hover {background-color: #9333ea;}
        </style>
        """, unsafe_allow_html=True)
    elif theme == "Forest Green":
        st.markdown("""
        <style>
            .main {background-color: #ecfdf5;}
            h1, h2, h3 {color: #065f46;}
            p {color: #064e3b;}
            .stButton>button {background-color: #10b981; color: white;}
            .stButton>button:hover {background-color: #059669;}
        </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
