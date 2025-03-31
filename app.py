import streamlit as st
import whisper
import tempfile
import os
from pydub import AudioSegment

# Initialize Whisper model globally to avoid reloading it multiple times
@st.cache_resource
def load_model():
    return whisper.load_model("base")

def transcribe_audio(audio_file_path):
    """Transcribe audio using Whisper."""
    model = load_model()
    result = model.transcribe(audio_file_path)
    return result["text"], result["segments"]

def generate_srt(segments, filename):
    """Generate SRT file from transcription segments."""
    srt_content = ""
    for i, segment in enumerate(segments):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"].strip()
        srt_content += f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n"
    
    srt_filename = f"{os.path.splitext(filename)[0]}.srt"
    with open(srt_filename, "w") as srt_file:
        srt_file.write(srt_content)
    return srt_filename

def format_timestamp(seconds):
    """Format timestamp for SRT (hh:mm:ss,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def main():
    st.set_page_config(page_title="Audio to Subtitles", layout="wide")
    
    # Header Section
    st.title("🎙️ Audio to Subtitles Converter")
    st.markdown("""
        Upload an audio file to transcribe it into text and generate subtitles (SRT format). 
        This app supports large audio files and provides real-time progress updates.
    """)

    # File Upload Section
    uploaded_file = st.file_uploader("Upload your audio file (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

    if uploaded_file is not None:
        # Display the uploaded audio file
        st.audio(uploaded_file, format="audio/wav", start_time=0)

        # Process the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Convert to WAV if necessary
        if uploaded_file.type != "audio/wav":
            audio = AudioSegment.from_file(tmp_file_path, format=uploaded_file.type.split("/")[1])
            wav_path = tmp_file_path + ".wav"
            audio.export(wav_path, format="wav")
        else:
            wav_path = tmp_file_path

        # Transcription Button
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing audio... This may take a while for large files."):
                transcription_text, segments = transcribe_audio(wav_path)
                st.subheader("Transcribed Text")
                st.text_area("Transcription", transcription_text, height=300)

                # Generate SRT File
                srt_file = generate_srt(segments, uploaded_file.name)

                # Provide Download Option for SRT File
                with open(srt_file, "r") as file:
                    srt_content = file.read()
                    st.download_button(
                        label="📥 Download Subtitles (SRT)",
                        data=srt_content,
                        file_name=os.path.basename(srt_file),
                        mime="text/plain"
                    )

                # Clean up temporary files
                os.unlink(tmp_file_path)
                if uploaded_file.type != "audio/wav":
                    os.unlink(wav_path)
                os.unlink(srt_file)

if __name__ == "__main__":
    main()
