import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile
import base64

# Set page configuration
st.set_page_config(
    page_title="Audio Fingerprinting System - Shazam Clone",
    page_icon="🎵",
    layout="wide"
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
    .info-box {
        padding: 1rem;
        background-color: #e0f2fe;
        border-left: 5px solid #3b82f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract simple audio features (mocked for demo)
def extract_features(audio_file):
    # In a real app, this would use audio processing libraries
    # For the demo, we'll use a hash of the file to generate consistent random features
    file_bytes = audio_file.getvalue()
    file_hash = hash(file_bytes) % 2**32
    np.random.seed(file_hash)
    
    # Generate random features but make them consistent for the same file
    return np.random.rand(25)

# Database of reference tracks
def get_reference_tracks():
    return {
        "Track 1 - Shape of You (Ed Sheeran)": {
            "artist": "Ed Sheeran",
            "album": "÷ (Divide)",
            "released": "2017",
            "genre": "Pop",
            "duration": "3:54"
        },
        "Track 2 - Blinding Lights (The Weeknd)": {
            "artist": "The Weeknd",
            "album": "After Hours",
            "released": "2020",
            "genre": "Synth-pop",
            "duration": "3:20"
        },
        "Track 3 - Uptown Funk (Mark Ronson ft. Bruno Mars)": {
            "artist": "Mark Ronson ft. Bruno Mars",
            "album": "Uptown Special",
            "released": "2015",
            "genre": "Funk",
            "duration": "4:30"
        },
        "Track 4 - Despacito (Luis Fonsi ft. Daddy Yankee)": {
            "artist": "Luis Fonsi ft. Daddy Yankee",
            "album": "Vida",
            "released": "2017",
            "genre": "Reggaeton",
            "duration": "3:47"
        },
        "Track 5 - Bad Guy (Billie Eilish)": {
            "artist": "Billie Eilish",
            "album": "When We All Fall Asleep, Where Do We Go?",
            "released": "2019",
            "genre": "Electropop",
            "duration": "3:14"
        }
    }

# Get reference features
def get_reference_features():
    np.random.seed(42)  # For reproducibility
    reference_features = {}
    
    for track in get_reference_tracks().keys():
        reference_features[track] = np.random.rand(25)
    
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
    
    # For demo purposes, return a match if similarity is above 0.6
    if highest_similarity > 0.6:
        return best_match, highest_similarity
    else:
        return None, highest_similarity

# Apply a selected theme
def apply_theme(theme_name):
    if theme_name == "Dark Blue":
        st.markdown("""
        <style>
            .main {background-color: #0f172a;}
            h1, h2, h3, p, label, .stSelectbox, table {color: #e2e8f0 !important;}
            .stButton>button {background-color: #3b82f6; color: white;}
            .stButton>button:hover {background-color: #2563eb;}
            div[data-testid="stVerticalBlock"] {background-color: #1e293b; padding: 1rem; border-radius: 0.5rem;}
        </style>
        """, unsafe_allow_html=True)
    elif theme_name == "Vibrant Purple":
        st.markdown("""
        <style>
            .main {background-color: #faf5ff;}
            h1, h2, h3 {color: #7e22ce;}
            p {color: #581c87;}
            .stButton>button {background-color: #a855f7; color: white;}
            .stButton>button:hover {background-color: #9333ea;}
        </style>
        """, unsafe_allow_html=True)
    elif theme_name == "Forest Green":
        st.markdown("""
        <style>
            .main {background-color: #ecfdf5;}
            h1, h2, h3 {color: #065f46;}
            p {color: #064e3b;}
            .stButton>button {background-color: #10b981; color: white;}
            .stButton>button:hover {background-color: #059669;}
        </style>
        """, unsafe_allow_html=True)

# Main app function
def main():
    st.title("🎵 Audio Fingerprinting System - Shazam Clone")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.markdown("""
    <div class="info-box">
    This app demonstrates a simplified version of audio fingerprinting, similar to how Shazam works.
    Upload an audio file to identify it against our database of reference tracks.
    </div>
    """, unsafe_allow_html=True)
    
    # Theme selector in sidebar
    st.sidebar.subheader("UI Theme")
    theme = st.sidebar.selectbox(
        "Select a theme",
        ["Default", "Dark Blue", "Vibrant Purple", "Forest Green"]
    )
    apply_theme(theme)
    
    # Main content area
    st.markdown("""
    <div class="info-box">
    <p>This application identifies songs from short audio clips, similar to apps like Shazam.</p>
    <p>Just upload an audio file below, and we'll try to identify it!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.subheader("Upload an audio sample to identify")
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "m4a"])
    
    if uploaded_file is not None:
        # Play the uploaded audio
        file_extension = os.path.splitext(uploaded_file.name)[1][1:].lower()
        st.audio(uploaded_file, format=f'audio/{file_extension}')
        
        # Identify button
        if st.button("Identify Song"):
            with st.spinner("Analyzing audio fingerprint..."):
                # Extract features
                query_features = extract_features(uploaded_file)
                
                # Get reference features
                reference_features = get_reference_features()
                
                # Identify the song
                match, confidence = identify_song(query_features, reference_features)
                
                # Display results
                if match:
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>🎵 Song Identified!</h3>
                    <p>We identified your audio as: <strong>{match}</strong></p>
                    <p>Confidence level: {confidence:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar for confidence
                    st.progress(float(confidence))
                    
                    # Get track details from our database
                    track_info = get_reference_tracks()[match]
                    
                    # Display track information in a table
                    st.subheader("Track Information")
                    data = {
                        "Property": list(track_info.keys()),
                        "Value": list(track_info.values())
                    }
                    st.table(pd.DataFrame(data))
                    
                else:
                    st.markdown("""
                    <div class="error-box">
                    <h3>❌ No match found</h3>
                    <p>We couldn't identify this audio clip in our database.</p>
                    <p>Try uploading a different section of the song or check if the audio is clear.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(f"Highest similarity score: {confidence:.2f}")
    
    # How it works section
    with st.expander("How Audio Fingerprinting Works"):
        st.markdown("""
        ### The Science Behind Shazam-like Applications
        
        Audio fingerprinting works by creating a unique "fingerprint" of audio that can be matched against a database:
        
        1. **Feature Extraction**: The app analyzes the audio spectrum and extracts key features
        2. **Fingerprint Creation**: These features are converted into a compact digital summary
        3. **Database Matching**: The fingerprint is compared against known songs in the database
        4. **Result Delivery**: The best match is returned, along with a confidence score
        
        This demo simulates this process using simplified algorithms for educational purposes.
        """)

if __name__ == "__main__":
    main()
