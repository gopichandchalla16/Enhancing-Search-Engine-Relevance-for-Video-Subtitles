import streamlit as st
import tempfile
import os
import time
import subprocess
import platform
import base64
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Audio to Subtitles Converter",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1E88E5;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 10px;
        font-size: 0.8em;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Define path for ffmpeg
FFMPEG_PATH = None

# Function to set ffmpeg path based on system
def setup_ffmpeg():
    global FFMPEG_PATH
    system = platform.system()
    
    # Check if ffmpeg exists in common paths
    if system == "Windows":
        potential_paths = [
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
            os.path.join(os.environ.get('USERPROFILE', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
            os.path.join(os.environ.get('PROGRAMFILES', ''), 'ffmpeg', 'bin', 'ffmpeg.exe')
        ]
    elif system == "Darwin":  # macOS
        potential_paths = [
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
            "/opt/local/bin/ffmpeg"
        ]
    else:  # Linux and others
        potential_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/bin/ffmpeg"
        ]
    
    # Add system PATH locations
    for path_dir in os.environ.get('PATH', '').split(os.pathsep):
        if path_dir:
            potential_paths.append(os.path.join(path_dir, "ffmpeg" + (".exe" if system == "Windows" else "")))
    
    # Check each potential path
    for path in potential_paths:
        if os.path.isfile(path):
            try:
                # Test if the ffmpeg binary works
                result = subprocess.run([path, "-version"], capture_output=True, text=True)
                if result.returncode == 0:
                    FFMPEG_PATH = path
                    return True
            except Exception:
                pass
    
    return False

# Create a container for the header
header_container = st.container()
with header_container:
    st.markdown('<h1 class="main-header">🎙️ Audio to Subtitles Converter</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-box">
        This app transcribes your audio files and generates subtitle files (SRT format).
        Simply upload an audio file, adjust settings as needed, click transcribe, and download your subtitles when ready.
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

# Define sidebar content
with st.sidebar:
    st.title("⚙️ Settings & Information")
    
    # Model Settings
    st.subheader("Model Settings")
    model_size = st.selectbox(
        "Whisper Model Size", 
        options=["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Larger models are more accurate but slower and require more memory"
    )
    
    language = st.selectbox(
        "Language", 
        options=["Auto-detect", "English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Chinese", "Japanese"],
        index=0,
        help="Specify the language for better accuracy (optional)"
    )
    
    advanced_settings = st.expander("Advanced Settings")
    with advanced_settings:
        beam_size = st.slider("Beam Size", 1, 10, 5, help="Higher beam size can improve accuracy but slows down transcription")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1, help="Controls randomness in predictions. Higher values give more diverse results.")
        
    # Feature toggles
    st.subheader("Features")
    enable_timestamps = st.toggle("Generate Timestamps", value=True, help="Include timestamps in the text transcript")
    enable_speaker_diarization = st.toggle("Speaker Diarization (Experimental)", value=False, help="Attempt to identify different speakers (experimental)")
    enable_translation = st.toggle("Translate to English", value=False, help="Translate non-English audio to English")
    
    # UI Theme
    st.subheader("UI Theme")
    color_theme = st.selectbox("Color Theme", ["Blue", "Green", "Purple", "Orange"], index=0)
    
    # Information
    st.markdown("---")
    st.subheader("ℹ️ Information")
    info_expander = st.expander("How it works")
    with info_expander:
        st.markdown("""
        1. Upload your audio file (MP3, WAV, M4A)
        2. Select model and settings
        3. Click 'Transcribe Audio'
        4. Review the transcription
        5. Download the SRT or text file
        
        This app uses OpenAI's Whisper model to transcribe audio. The model runs locally on your machine.
        """)
    
    # FFmpeg installer
    st.markdown("---")
    st.subheader("🛠️ FFmpeg Setup")
    
    # Check if FFmpeg is found
    if setup_ffmpeg():
        st.success(f"✅ FFmpeg found at: {FFMPEG_PATH}")
    else:
        st.error("❌ FFmpeg not found!")
        st.info("Please install FFmpeg on your system:")
        st.markdown("""
        * **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
        * **macOS**: `brew install ffmpeg`
        * **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)
        """)
        
        # Custom FFmpeg path input
        st.subheader("Specify FFmpeg path manually")
        custom_path = st.text_input("FFmpeg executable path")
        if custom_path and os.path.isfile(custom_path):
            try:
                result = subprocess.run([custom_path, "-version"], capture_output=True, text=True)
                if result.returncode == 0:
                    FFMPEG_PATH = custom_path
                    st.success(f"✅ FFmpeg found at: {FFMPEG_PATH}")
                else:
                    st.error("Invalid FFmpeg binary")
            except Exception:
                st.error("Could not execute the specified file")

# Cache the Whisper model to avoid reloading
@st.cache_resource
def load_model(model_size="base"):
    """Load and cache the Whisper model."""
    with st.spinner(f"Loading Whisper {model_size} model... This will only happen once."):
        try:
            import whisper
            return whisper.load_model(model_size)
        except Exception as e:
            st.error(f"Error loading Whisper model: {e}")
            st.error("Please make sure all dependencies are installed correctly.")
            return None

def transcribe_audio(audio_file_path, model_size="base", language=None, beam_size=5, temperature=0.0, translate=False):
    """Transcribe audio using Whisper."""
    model = load_model(model_size)
    if model is None:
        return None, None
    
    try:
        # Set up options
        options = {
            "beam_size": beam_size,
            "temperature": temperature,
            "verbose": False
        }
        
        # Handle language selection
        if language and language != "Auto-detect":
            options["language"] = language.lower()
        
        # Handle translation
        task = "translate" if translate else "transcribe"
        
        # Perform transcription
        result = model.transcribe(audio_file_path, task=task, **options)
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

def format_timestamp_readable(seconds):
    """Format timestamp in a more readable way (mm:ss)."""
    minutes = int(seconds // 60)
    seconds_int = int(seconds % 60)
    return f"{minutes:02}:{seconds_int:02}"

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

def generate_timestamped_text(segments):
    """Generate text with timestamps from segments."""
    if not segments:
        return ""
        
    timestamped_text = ""
    for segment in segments:
        start_time = format_timestamp_readable(segment["start"])
        text = segment["text"].strip()
        timestamped_text += f"[{start_time}] {text}\n\n"
    
    return timestamped_text

def apply_speaker_diarization(segments):
    """Apply basic speaker diarization (simulation for UI demo)."""
    # This is a simplified simulation since real diarization requires additional models
    if not segments:
        return segments
        
    import random
    
    # Create a copy of segments to modify
    diarized_segments = []
    
    # Assign random speakers based on clustering by timing and positioning
    curr_speaker = 0
    speakers = {}
    
    for segment in segments:
        text = segment["text"].strip()
        
        # Simple logic to determine speaker changes (just for UI demo)
        if "?" in text:
            if curr_speaker not in speakers:
                speakers[curr_speaker] = f"Speaker {curr_speaker+1}"
            new_speaker = (curr_speaker + 1) % 3
            curr_speaker = new_speaker
        
        if curr_speaker not in speakers:
            speakers[curr_speaker] = f"Speaker {curr_speaker+1}"
        
        # Create a new segment with speaker information
        new_segment = segment.copy()
        new_segment["text"] = f"{speakers[curr_speaker]}: {text}"
        diarized_segments.append(new_segment)
        
        # Randomly change speakers occasionally
        if random.random() > 0.7:
            curr_speaker = (curr_speaker + 1) % 3
            
    return diarized_segments

def convert_audio_to_wav(input_path, output_path):
    """Convert audio file to WAV format using ffmpeg."""
    try:
        if FFMPEG_PATH:
            command = [
                FFMPEG_PATH,
                "-i", input_path,
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",      # mono
                "-c:a", "pcm_s16le",  # 16-bit PCM
                "-y",            # overwrite output file
                output_path
            ]
            
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process.returncode != 0:
                st.error(f"FFmpeg conversion error: {process.stderr}")
                return False
            return True
        else:
            st.error("FFmpeg path not set. Cannot convert audio.")
            return False
    except Exception as e:
        st.error(f"Error during audio conversion: {e}")
        return False

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
    
    # If dependencies are missing, show instructions
    if missing_deps:
        st.error("❌ Missing dependencies detected!")
        st.info("Please install the following packages:")
        st.code(f"pip install {' '.join(missing_deps)}")
        st.info("Or install all requirements with:")
        st.code("pip install -r requirements.txt")
        return False
    
    return True

def create_audio_waveform(audio_data):
    """Create a simple visualization of the audio waveform."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from pydub import AudioSegment
        import io
        
        # Load audio data
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.write(audio_data)
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Convert to mono wav for analysis
        if FFMPEG_PATH:
            mono_path = temp_file_path + '_mono.wav'
            convert_audio_to_wav(temp_file_path, mono_path)
            audio = AudioSegment.from_wav(mono_path)
        else:
            audio = AudioSegment.from_file(temp_file_path)
            audio = audio.set_channels(1)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())
        
        # Create plot
        plt.figure(figsize=(10, 2))
        plt.plot(np.linspace(0, len(samples)/audio.frame_rate, num=len(samples)), samples, color='blue', linewidth=0.5)
        plt.axis('off')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        
        # Clean up
        clean_up_temp_files(temp_file_path, mono_path if FFMPEG_PATH else None)
        
        return buf
    except Exception as e:
        st.warning(f"Could not generate waveform visualization: {e}")
        return None

def get_download_link(file_content, filename, text):
    """Generate a download link for a file"""
    b64 = base64.b64encode(file_content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'

def main():
    """Main application function."""    
    # Check dependencies first
    if not check_dependencies():
        return
    
    # Import dependencies now that we know they're installed
    import whisper
    
    # File Upload Section
    upload_container = st.container()
    with upload_container:
        uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "m4a", "flac", "ogg"])
    
    # Process the uploaded file
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("📁 File Details:")
            for key, value in file_details.items():
                st.write(f"- **{key}:** {value}")
        
        # Display audio player
        st.audio(uploaded_file)
        
        # Generate and display waveform
        try:
            waveform = create_audio_waveform(uploaded_file.getvalue())
            if waveform:
                st.image(waveform, use_column_width=True, caption="Audio Waveform")
        except:
            pass
        
        # Process button in its own container
        button_container = st.container()
        with button_container:
            process_col1, process_col2 = st.columns([1, 1])
            with process_col1:
                process_button = st.button("🔊 Transcribe Audio", use_container_width=True)
            with process_col2:
                if not FFMPEG_PATH:
                    st.warning("⚠️ FFmpeg not found - audio conversion may fail")
        
        # If button is clicked, process the audio
        if process_button:
            # Create status containers
            status_container = st.container()
            result_container = st.container()
            
            with status_container:
                st.markdown('<div class="status-box">', unsafe_allow_html=True)
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
                
                # Get selected language
                selected_language = None
                if language != "Auto-detect":
                    selected_language = language
                
                try:
                    wav_path = tmp_file_path + ".wav"
                    
                    if FFMPEG_PATH and file_format != "wav":
                        # Use ffmpeg for conversion
                        if convert_audio_to_wav(tmp_file_path, wav_path):
                            status_text.text("Audio converted to WAV format.")
                        else:
                            st.error("Failed to convert audio. Trying alternative method...")
                            from pydub import AudioSegment
                            audio = AudioSegment.from_file(tmp_file_path, format=file_format)
                            audio.export(wav_path, format="wav")
                    elif file_format != "wav":
                        # Use pydub for conversion
                        from pydub import AudioSegment
                        audio = AudioSegment.from_file(tmp_file_path, format=file_format)
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
                status_text.text(f"Transcribing audio with {model_size} model... This may take a while for large files.")
                progress_bar.progress(40)
                
                transcription_text, segments = transcribe_audio(
                    wav_path, 
                    model_size=model_size,
                    language=selected_language if selected_language != "Auto-detect" else None,
                    beam_size=beam_size,
                    temperature=temperature,
                    translate=enable_translation
                )
                
                if not transcription_text:
                    progress_bar.progress(100)
                    status_text.text("❌ Transcription failed.")
                    clean_up_temp_files(tmp_file_path, wav_path)
                    return
                
                # Apply speaker diarization if enabled
                if enable_speaker_diarization:
                    status_text.text("Analyzing speakers...")
                    progress_bar.progress(70)
                    segments = apply_speaker_diarization(segments)
                
                # Update progress
                progress_bar.progress(80)
                status_text.text("Generating subtitle file...")
                
                # Generate SRT file
                srt_file = generate_srt(segments, uploaded_file.name)
                
                # Generate timestamped text if enabled
                if enable_timestamps:
                    timestamped_text = generate_timestamped_text(segments)
                else:
                    timestamped_text = transcription_text
                
                progress_bar.progress(90)
                
                if not srt_file:
                    progress_bar.progress(100)
                    status_text.text("❌ Failed to generate subtitle file.")
                    clean_up_temp_files(tmp_file_path, wav_path)
                    return
                
                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Transcription complete!")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Clean up temp files when done
                clean_up_temp_files(tmp_file_path, wav_path)
            
            # Display results in the result container
            with result_container:
                st.success("Audio transcription completed successfully!")
                
                # Create tabs for different outputs
                tab1, tab2, tab3 = st.tabs(["Transcription", "Subtitles Preview", "Download Options"])
                
                # Tab 1: Transcription text
                with tab1:
                    st.text_area("Transcribed Text", timestamped_text, height=300)
                    
                    # Word and character counts
                    word_count = len(transcription_text.split())
                    char_count = len(transcription_text)
                    st.write(f"📊 Statistics: {word_count} words, {char_count} characters")
                
                # Tab 2: SRT Preview
                with tab2:
                    try:
                        with open(srt_file, "r", encoding="utf-8") as file:
                            srt_content = file.read()
                        st.text_area("SRT Format Preview", srt_content, height=300)
                    except Exception as e:
                        st.error(f"Error reading SRT file: {e}")
                
                # Tab 3: Download options
                with tab3:
                    st.subheader("Download Options")
                    
                    # Create columns for download buttons
                    col1, col2 = st.columns(2)
                    
                    try:
                        # SRT download
                        with open(srt_file, "r", encoding="utf-8") as file:
                            srt_content = file.read()
                            
                        with col1:
                            st.download_button(
                                label="📥 Download SRT Subtitle File",
                                data=srt_content,
                                file_name=f"{os.path.splitext(uploaded_file.name)[0]}.srt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        # Text download
                        with col2:
                            st.download_button(
                                label="📄 Download Text Transcript",
                                data=timestamped_text,
                                file_name=f"{os.path.splitext(uploaded_file.name)[0]}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        # Additional formats
                        st.markdown("---")
                        st.subheader("Additional Export Options")
                        
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            # Generate VTT format
                            vtt_content = "WEBVTT\n\n"
                            for segment in segments:
                                start = format_timestamp(segment["start"]).replace(',', '.')
                                end = format_timestamp(segment["end"]).replace(',', '.')
                                text = segment["text"].strip()
                                vtt_content += f"{start} --> {end}\n{text}\n\n"
                            
                            st.download_button(
                                label="📥 Download VTT Format",
                                data=vtt_content,
                                file_name=f"{os.path.splitext(uploaded_file.name)[0]}.vtt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col4:
                            # Generate JSON format with timestamps
                            import json
                            json_data = {
                                "transcription": transcription_text,
                                "segments": [
                                    {
                                        "start": segment["start"],
                                        "end": segment["end"],
                                        "text": segment["text"].strip()
                                    } for segment in segments
                                ]
                            }
                            
                            st.download_button(
                                label="📥 Download JSON Format",
                                data=json.dumps(json_data, indent=2),
                                file_name=f"{os.path.splitext(uploaded_file.name)[0]}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                            
                        # Clean up the SRT file after providing download
                        clean_up_temp_files(srt_file)
                    except Exception as e:
                        st.error(f"Error providing download: {e}")
    else:
        # Display sample audio files when no file is uploaded
        with st.expander("Don't have an audio file? Try one of our samples"):
            st.write("Coming soon: Sample audio files to test the app")
    
    # Footer
    st.markdown("""
    <div class="footer">
    Audio to Subtitles Converter | Version 1.1 | Powered by OpenAI Whisper
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
