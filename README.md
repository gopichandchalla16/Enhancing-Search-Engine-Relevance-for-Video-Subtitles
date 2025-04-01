Enhancing Search Engine Relevance for Video Subtitles

Welcome to the Enhancing Search Engine Relevance for Video Subtitles project! This repository contains a Streamlit-based web application designed to convert audio files into subtitle files (SRT format) and enhance search engine relevance for video content. The app leverages advanced audio processing and transcription techniques, with additional utilities for chunking, cleaning, and embedding subtitles for improved searchability.

Project Overview

The primary goal of this project is to:

Convert audio files (MP3, WAV, M4A) into accurate SRT subtitles using OpenAI's Whisper model.
Enhance the searchability of video content by processing subtitles for relevance (e.g., cleaning, chunking, and embedding).
Provide a user-friendly interface with features like audio playback, transcription editing, and multiple export options.

Key features include:

Audio-to-Subtitle Conversion: Transcribe audio with customizable model sizes and languages.

Noise Reduction: Optional basic noise reduction for clearer transcription.

Editable Transcriptions: Modify transcriptions before generating subtitles.

Export Options: Download subtitles as SRT, TXT, or JSON.

Progress Tracking: Visual feedback during processing.

Themed UI: Attractive, customizable interface with Light, Dark, and Forest themes.

Repository Structure

app.py: Main Streamlit application for audio-to-subtitle conversion.

requirements.txt: List of Python dependencies required to run the app.

audio_to_subtitles.ipynb: Jupyter notebook exploring the audio-to-subtitle pipeline.

chunking.ipynb: Notebook for splitting subtitles into searchable chunks.

cleaning_subtitle.ipynb: Notebook for cleaning subtitle data to improve quality.

embedding_chromoDB.ipynb: Notebook for embedding subtitles into a ChromaDB database for search enhancement.

Prerequisites

Python: Version 3.8 or higher.

Streamlit Cloud: Optional, for deployment (no FFmpeg required).

Git: For cloning the repository.

Using the App

Upload Audio: Drag and drop an audio file (e.g., "i_hear_voices.mp3") into the uploader.

Configure Settings:

Select model size (tiny, base, small).

Choose language (Auto-detect, English, Spanish).

Adjust confidence threshold and enable noise reduction if needed.

Convert: Click "Convert to Subtitles" to process the audio.

Interact:

Play the audio using the built-in player.

Edit the transcription if desired.

Download results as SRT, TXT, or JSON.

Streamlit URL : https://searchmysubtitles.streamlit.app/ 
