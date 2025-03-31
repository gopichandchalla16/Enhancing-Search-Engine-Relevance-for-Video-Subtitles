import streamlit as st
import tempfile
import os
import time

# Cache the Whisper model to avoid reloading
@st.cache_resource
def load_model():
    try:
        import whisper
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

def transcribe_audio(audio_file_path):
    """Transcribe audio using Whisper."""
    model = load_model()
    if model is None:
        st.error("Transcription model could not be loaded.")
        return None, None
    
    try:
        result = model.transcribe(audio_file_path)
        return result["text"], result["segments"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None, None

def generate_srt(segments, filename):
    """Generate SRT file from transcription segments."""
    srt_content = ""
    for i, segment in enumerate(segments):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"].strip()
        srt_content += f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n"
    
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), "streamlit_srt")
    os.makedirs(temp_dir, exist_ok=True)
    
    srt_filename = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(filename))[0]}.srt")
    try:
        with open(srt_filename, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)
        return srt_filename
    except Exception as e:
        st.error(f"Error generating SRT file: {e}")
        return None

def format_timestamp(seconds):
    """Format timestamp for SRT (hh:mm:ss,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"

def clean_up_temp_files(*files):
    """Clean up temporary files."""
    for file_path in files:
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            pass

def main():
    st.set_page_config(page_title="Audio to Subtitles", layout="wide")
    
    # Header Section
    st.title("🎙️ Audio to Subtitles Converter")
    st.markdown("""
        Upload an audio file to transcribe it into text and generate subtitles (SRT format).
        This app supports various audio formats and provides real-time progress updates.
    """)

    # Import dependencies
    try:
        import whisper
        from pydub import AudioSegment
    except ImportError:
        st.error("Required dependencies are missing. Please install them using:")
        st.code("pip install -r requirements.txt")
        st.markdown("Note: You also need to install ffmpeg on your system:")
        st.markdown("- **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`")
        st.markdown("- **macOS**: `brew install ffmpeg`")
        st.markdown("- **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)")
        st.stop()

    # File Upload Section
    uploaded_file = st.file_uploader("Upload your audio file (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

    if uploaded_file is not None:
        # Display the uploaded audio file
        st.audio(uploaded_file)

        # Create a progress placeholder
        progress_placeholder = st.empty()
        
        # Process the uploaded file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
        except Exception as e:
            st.error(f"Error creating temporary file: {e}")
            return

        # Determine file format
        file_format = uploaded_file.name.split('.')[-1].lower()
        
        # Convert to WAV if necessary
        try:
            progress_placeholder.text("Processing audio file...")
            if file_format != "wav":
                audio = AudioSegment.from_file(tmp_file_path, format=file_format)
                wav_path = tmp_file_path + ".wav"
                audio.export(wav_path, format="wav")
                progress_placeholder.text("Audio converted to WAV format successfully.")
            else:
                wav_path = tmp_file_path
                progress_placeholder.text("Audio file ready for transcription.")
        except Exception as e:
            st.error(f"Error processing audio file: {e}")
            st.error("Make sure ffmpeg is properly installed on your system.")
            clean_up_temp_files(tmp_file_path)
            return

        # Transcription Button
        if st.button("Transcribe Audio"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Loading model phase
            status_text.text("Loading transcription model...")
            for i in range(10):
                time.sleep(0.1)
                progress_bar.progress(i * 0.05)
            
            # Transcription phase
            status_text.text("Transcribing audio... This may take a while for large files.")
            transcription_text, segments = transcribe_audio(wav_path)
            
            if transcription_text:
                # Complete progress
                for i in range(20, 100):
                    progress_bar.progress(i * 0.01)
                    time.sleep(0.01)
                
                progress_bar.progress(100)
                status_text.text("Transcription completed successfully!")
                
                st.subheader("Transcribed Text")
                st.text_area("Transcription", transcription_text, height=300)

                # Generate SRT File
                status_text.text("Generating subtitle file...")
                srt_file = generate_srt(segments, uploaded_file.name)

                if srt_file:
                    # Provide Download Option for SRT File
                    try:
                        with open(srt_file, "r", encoding="utf-8") as file:
                            srt_content = file.read()
                            st.download_button(
                                label="📥 Download Subtitles (SRT)",
                                data=srt_content,
                                file_name=os.path.basename(srt_file),
                                mime="text/plain"
                            )
                        status_text.text("✅ Process complete! You can now download your subtitle file.")
                    except Exception as e:
                        st.error(f"Error providing SRT file download: {e}")
                    finally:
                        try:
                            os.remove(srt_file)
                        except:
                            pass
                else:
                    st.error("Failed to generate SRT file.")
            else:
                progress_bar.progress(100)
                status_text.text("❌ Transcription failed.")
                st.error("Failed to transcribe audio. Please check if the audio file is valid and try again.")
            
            # Clean up temporary files
            clean_up_temp_files(tmp_file_path, wav_path)

if __name__ == "__main__":
    main()
