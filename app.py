import streamlit as st
import tempfile
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Audio to Subtitles Converter",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create a container for the header
header_container = st.container()
with header_container:
    st.title("🎙️ Audio to Subtitles Converter")
    st.markdown("""
        This app transcribes your audio files and generates subtitle files (SRT format).
        Simply upload an audio file, click transcribe, and download your subtitles when ready.
    """)
    st.markdown("---")

# Define sidebar content
with st.sidebar:
    st.title("ℹ️ Information")
    st.markdown("""
    ### How it works
    1. Upload your audio file (MP3, WAV, M4A)
    2. Click 'Transcribe Audio'
    3. View the transcription text
    4. Download the SRT subtitle file
    
    ### About
    This app uses OpenAI's Whisper model to transcribe audio. The model will be downloaded automatically on first use.
    
    ### Requirements
    - Python 3.7+
    - FFmpeg installed on your system
    """)
    
    with st.expander("System Requirements"):
        st.markdown("""
        #### FFmpeg Installation:
        - **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
        - **macOS**: `brew install ffmpeg`
        - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)
        """)

# Cache the Whisper model to avoid reloading
@st.cache_resource
def load_model():
    """Load and cache the Whisper model."""
    with st.spinner("Loading Whisper model... This will only happen once."):
        try:
            import whisper
            return whisper.load_model("base")
        except Exception as e:
            st.error(f"Error loading Whisper model: {e}")
            st.error("Please make sure all dependencies are installed correctly.")
            return None

def transcribe_audio(audio_file_path):
    """Transcribe audio using Whisper."""
    model = load_model()
    if model is None:
        return None, None
    
    try:
        result = model.transcribe(audio_file_path)
        return result["text"], result["segments"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None, None

def format_timestamp(seconds):
    """Format timestamp for SRT (hh:mm:ss,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"

def generate_srt(segments, filename):
    """Generate SRT file from transcription segments."""
    if not segments:
        return None
        
    srt_content = ""
    for i, segment in enumerate(segments):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"].strip()
        srt_content += f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n"
    
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), "streamlit_srt")
    os.makedirs(temp_dir, exist_ok=True)
    
    base_filename = os.path.basename(filename)
    srt_filename = os.path.join(temp_dir, f"{os.path.splitext(base_filename)[0]}.srt")
    
    try:
        with open(srt_filename, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)
        return srt_filename
    except Exception as e:
        st.error(f"Error generating SRT file: {e}")
        return None

def clean_up_temp_files(*files):
    """Clean up temporary files."""
    for file_path in files:
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except Exception:
            pass  # Silently continue if cleanup fails

def check_dependencies():
    """Check if dependencies are installed."""
    missing_deps = []
    
    # Check Python dependencies
    try:
        import whisper
    except ImportError:
        missing_deps.append("openai-whisper")
    
    try:
        from pydub import AudioSegment
    except ImportError:
        missing_deps.append("pydub")
    
    # If dependencies are missing, show instructions
    if missing_deps:
        st.error("❌ Missing dependencies detected!")
        st.info("Please install the following packages:")
        st.code(f"pip install {' '.join(missing_deps)}")
        st.info("Or install all requirements with:")
        st.code("pip install -r requirements.txt")
        return False
    
    # Check if ffmpeg is installed
    try:
        import subprocess
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        st.error("❌ FFmpeg not found!")
        st.info("Please install FFmpeg on your system:")
        st.markdown("""
        - **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
        - **macOS**: `brew install ffmpeg`
        - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)
        """)
        return False
    
    return True

def main():
    """Main application function."""    
    # Check dependencies first
    if not check_dependencies():
        return
    
    # Import dependencies now that we know they're installed
    import whisper
    from pydub import AudioSegment
    
    # File Upload Section
    upload_container = st.container()
    with upload_container:
        uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "m4a"])
    
    # Process the uploaded file
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        st.write("File Details:")
        for key, value in file_details.items():
            st.write(f"- {key}: {value}")
        
        # Display the uploaded audio file
        st.audio(uploaded_file)
        
        # Process button in its own container
        button_container = st.container()
        with button_container:
            process_col, _ = st.columns([1, 2])
            with process_col:
                process_button = st.button("🔊 Transcribe Audio", use_container_width=True)
        
        # If button is clicked, process the audio
        if process_button:
            # Create status containers
            status_container = st.container()
            result_container = st.container()
            
            with status_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Save uploaded file to temp location
                status_text.text("Saving uploaded file...")
                progress_bar.progress(10)
                
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                except Exception as e:
                    st.error(f"Error saving uploaded file: {e}")
                    return
                
                # Determine file format
                file_format = uploaded_file.name.split('.')[-1].lower()
                
                # Convert to WAV if necessary
                status_text.text("Processing audio file...")
                progress_bar.progress(20)
                
                try:
                    if file_format != "wav":
                        audio = AudioSegment.from_file(tmp_file_path, format=file_format)
                        wav_path = tmp_file_path + ".wav"
                        audio.export(wav_path, format="wav")
                        status_text.text("Audio converted to WAV format.")
                    else:
                        wav_path = tmp_file_path
                    progress_bar.progress(30)
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                    st.error("Make sure FFmpeg is properly installed.")
                    clean_up_temp_files(tmp_file_path)
                    return
                
                # Transcribe audio
                status_text.text("Transcribing audio... This may take a while for large files.")
                progress_bar.progress(40)
                
                transcription_text, segments = transcribe_audio(wav_path)
                
                if not transcription_text:
                    progress_bar.progress(100)
                    status_text.text("❌ Transcription failed.")
                    clean_up_temp_files(tmp_file_path, wav_path)
                    return
                
                # Update progress
                progress_bar.progress(80)
                status_text.text("Generating subtitle file...")
                
                # Generate SRT file
                srt_file = generate_srt(segments, uploaded_file.name)
                progress_bar.progress(90)
                
                if not srt_file:
                    progress_bar.progress(100)
                    status_text.text("❌ Failed to generate subtitle file.")
                    clean_up_temp_files(tmp_file_path, wav_path)
                    return
                
                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Transcription complete!")
                
                # Clean up temp files when done
                clean_up_temp_files(tmp_file_path, wav_path)
            
            # Display results in the result container
            with result_container:
                st.success("Audio transcription completed successfully!")
                
                # Display transcription text
                st.subheader("📝 Transcription")
                with st.expander("View transcribed text", expanded=True):
                    st.text_area("", transcription_text, height=200)
                
                # Download options
                st.subheader("📥 Download")
                
                # Subtitle file download
                try:
                    with open(srt_file, "r", encoding="utf-8") as file:
                        srt_content = file.read()
                        
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download SRT Subtitle File",
                            data=srt_content,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}.srt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download Text Transcript",
                            data=transcription_text,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                    # Clean up the SRT file after providing download
                    clean_up_temp_files(srt_file)
                except Exception as e:
                    st.error(f"Error providing download: {e}")

if __name__ == "__main__":
    main()
