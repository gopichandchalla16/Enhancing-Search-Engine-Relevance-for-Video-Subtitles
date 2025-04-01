import streamlit as st
import tempfile
import os
from pathlib import Path
import whisper
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io

# Page config
st.set_page_config(
    page_title="Audio to Subtitles",
    page_icon="🎙️",
    layout="centered"
)

# Theme options
THEMES = {
    "Light": {"primary": "#3498db", "background": "#ffffff", "text": "#2c3e50", "secondary": "#e8f4f8"},
    "Dark": {"primary": "#2980b9", "background": "#2c3e50", "text": "#ecf0f1", "secondary": "#34495e"},
    "Forest": {"primary": "#27ae60", "background": "#f1f8e9", "text": "#2d572c", "secondary": "#dcedc8"}
}

# Enhanced Custom CSS
def apply_theme(theme):
    css = f"""
    <style>
    /* Global styles */
    body {{ 
        background-color: {theme['background']}; 
        font-family: 'Poppins', sans-serif; 
    }}
    
    /* Title and subtitle */
    .title {{ 
        color: {theme['text']}; 
        font-size: 2.8em; 
        text-align: center; 
        font-weight: 700; 
        margin-top: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}
    .subtitle {{ 
        color: {theme['text']}; 
        text-align: center; 
        opacity: 0.8; 
        font-size: 1.2em; 
        margin-bottom: 30px;
    }}

    /* File uploader */
    .stFileUploader {{
        border: 2px dashed {theme['primary']};
        border-radius: 10px;
        padding: 20px;
        background-color: {theme['secondary']};
        transition: all 0.3s ease;
    }}
    .stFileUploader:hover {{
        border-color: {theme['text']};
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}

    /* Buttons */
    .stButton>button {{ 
        background: linear-gradient(45deg, {theme['primary']}, {theme['text']}); 
        color: white; 
        width: 100%; 
        border-radius: 8px; 
        padding: 12px 24px;
        font-size: 1.1em;
        font-weight: 600;
        border: none;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .stButton>button:hover {{ 
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }}

    /* Success box */
    .success-box {{ 
        background-color: {theme['secondary']}; 
        padding: 20px; 
        border-radius: 12px; 
        margin-top: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid {theme['primary']};
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 20px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {theme['secondary']};
        border-radius: 8px;
        padding: 10px 20px;
        color: {theme['text']};
        transition: all 0.3s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {theme['primary']};
        color: white;
    }}

    /* Sidebar */
    .css-1d391kg {{
        background-color: {theme['secondary']};
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }}
    .stSlider > div > div > div > div {{
        background-color: {theme['primary']};
    }}

    /* Footer */
    footer {{
        text-align: center; 
        padding: 20px; 
        color: {theme['text']}; 
        opacity: 0.7;
        font-size: 0.9em;
    }}

    /* Waveform */
    .stImage {{
        background-color: {theme['secondary']};
        padding: 10px;
        border-radius: 8px;
        margin: 20px 0;
    }}

    /* Select boxes */
    .stSelectbox > div > div {{
        background-color: {theme['secondary']};
        border-radius: 6px;
        padding: 8px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    # Add Google Fonts for better typography
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# Core functions (unchanged)
@st.cache_resource
def load_model(model_size="base"):
    return whisper.load_model(model_size)

def load_audio_to_array(audio_bytes, original_ext):
    """Load audio to numpy array using librosa"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        # Load audio with librosa
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        return y, sr, tmp_path
    except Exception as e:
        st.error(f"Audio loading failed: {e}")
        return None, None, tmp_path

def format_timestamp(seconds):
    ms = int((seconds % 1) * 1000)
    seconds = int(seconds)
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def generate_srt(segments, filename):
    srt_content = ""
    for i, segment in enumerate(segments):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        srt_content += f"{i+1}\n{start} --> {end}\n{text}\n\n"
    
    srt_path = os.path.join(tempfile.gettempdir(), f"{Path(filename).stem}.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return srt_path

def generate_waveform(audio_data, sr):
    try:
        plt.figure(figsize=(10, 2))
        plt.plot(np.linspace(0, len(audio_data)/sr, len(audio_data)), audio_data, color=THEMES[st.session_state.theme]["primary"])
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
        plt.close()
        buf.seek(0)
        return buf
    except:
        return None

# Main app
def main():
    # Theme selection
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"
    theme = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=0)
    st.session_state.theme = theme
    apply_theme(THEMES[theme])

    st.markdown('<div class="title">🎙️ Audio to Subtitles</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Convert your audio files to SRT subtitles</div>', unsafe_allow_html=True)

    # Sidebar features
    with st.sidebar:
        st.header("Options")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)

    # File uploader
    audio_file = st.file_uploader(
        "Upload Audio File",
        type=["mp3", "wav", "m4a"],
        help="Limit 200MB per file"
    )

    if audio_file:
        st.write(f"📄 **File:** {audio_file.name} ({audio_file.size / 1024:.1f}KB)")

        # Settings
        col1, col2 = st.columns(2)
        with col1:
            model_size = st.selectbox("Model Size", ["tiny", "base", "small"], index=1)
        with col2:
            language = st.selectbox("Language", ["Auto-detect", "English", "Spanish"], index=0)

        # Transcribe button
        if st.button("Convert to Subtitles"):
            with st.spinner("Processing audio..."):
                # Load audio directly to array
                audio_data, sr, tmp_path = load_audio_to_array(audio_file.read(), Path(audio_file.name).suffix)
                if audio_data is None:
                    os.unlink(tmp_path)
                    return

                # Show waveform
                waveform = generate_waveform(audio_data, sr)
                if waveform:
                    st.image(waveform, caption="Audio Waveform")

                # Transcribe directly with audio array
                model = load_model(model_size)
                lang = language if language != "Auto-detect" else None
                result = model.transcribe(audio_data, language=lang)
                
                # Filter by confidence
                segments = [seg for seg in result["segments"] if seg.get("confidence", 1.0) >= confidence_threshold]
                
                # Generate SRT
                srt_path = generate_srt(segments, audio_file.name)

                # Display results
                with st.container():
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success("Conversion Complete!")
                    
                    tabs = st.tabs(["Text", "SRT Preview", "Download"])
                    
                    with tabs[0]:
                        st.text_area("Transcription", result["text"], height=200)
                    
                    with tabs[1]:
                        with open(srt_path, "r", encoding="utf-8") as f:
                            st.text_area("SRT Preview", f.read(), height=200)
                    
                    with tabs[2]:
                        with open(srt_path, "rb") as f:
                            st.download_button(
                                "Download SRT",
                                f.read(),
                                f"{Path(audio_file.name).stem}.srt",
                                "text/plain"
                            )
                    st.markdown('</div>', unsafe_allow_html=True)

                # Cleanup
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                if os.path.exists(srt_path):
                    os.unlink(srt_path)

    st.markdown("<footer style='text-align: center; padding: 20px;'>Made with Streamlit</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
