import streamlit as st
import tempfile
import os
import subprocess
from pathlib import Path
import whisper

# Page config
st.set_page_config(
    page_title="Audio to Subtitles Converter",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .title { color: #2c3e50; font-size: 2.5em; text-align: center; }
    .subtitle { color: #7f8c8d; text-align: center; }
    .stButton>button { 
        background-color: #3498db; 
        color: white; 
        width: 100%;
        border-radius: 5px;
    }
    .success-box { 
        background-color: #e8f4f8; 
        padding: 10px; 
        border-radius: 5px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# FFmpeg setup
def setup_ffmpeg():
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

# Core functions
@st.cache_resource
def load_model(model_size="base"):
    return whisper.load_model(model_size)

def convert_to_wav(input_path):
    wav_path = input_path + ".wav"
    cmd = ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", wav_path, "-y"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return wav_path if result.returncode == 0 else None

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

# Main app
def main():
    st.markdown('<div class="title">🎙️ Audio to Subtitles</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Convert your audio files to SRT subtitles</div>', unsafe_allow_html=True)

    # FFmpeg check
    if not setup_ffmpeg():
        st.error("FFmpeg not found! Please install it:\n- Ubuntu: `sudo apt install ffmpeg`\n- macOS: `brew install ffmpeg`\n- Windows: Download from ffmpeg.org")
        return

    # File uploader
    audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])

    if audio_file:
        # Settings
        col1, col2 = st.columns(2)
        with col1:
            model_size = st.selectbox("Model Size", ["tiny", "base", "small", "medium", "large"], index=1)
        with col2:
            language = st.selectbox("Language", ["Auto-detect", "English", "Spanish", "French"], index=0)

        # Transcribe button
        if st.button("Convert to Subtitles"):
            with st.spinner("Processing..."):
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name

                # Convert to WAV
                wav_path = convert_to_wav(tmp_path)
                if not wav_path:
                    st.error("Audio conversion failed!")
                    os.unlink(tmp_path)
                    return

                # Transcribe
                model = load_model(model_size)
                lang = language if language != "Auto-detect" else None
                result = model.transcribe(wav_path, language=lang)
                
                # Generate SRT
                srt_path = generate_srt(result["segments"], audio_file.name)

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
                for path in [tmp_path, wav_path, srt_path]:
                    if os.path.exists(path):
                        os.unlink(path)

if __name__ == "__main__":
    main()
