# Speech Emotion Recognition with Whisper-Large-V3

A Streamlit-based web application for real-time speech emotion recognition using the Whisper-Large-V3 model fine-tuned for emotion detection. The application supports both audio file uploads and live microphone recording.

## Overview

This application uses a pre-trained Hugging Face model to classify emotions in speech audio. It can detect eight different emotions: angry, calm, disgust, fear, happy, neutral, sad, and surprise.

## Features

- Upload multiple audio files for batch emotion analysis
- Record audio directly from your microphone
- Real-time emotion prediction with confidence scores
- Probability distribution visualization for all emotion classes
- Summary statistics and emotion distribution charts
- Support for multiple audio formats (WAV, MP3, FLAC, OGG)

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Microphone access (for recording features)
- CUDA-compatible GPU (optional, for faster inference)

## Installation

### Step 1: Clone or Download the Repository

```bash
git clone <repository-url>
cd <project-directory>
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### Step 4: Install Required Dependencies

```bash
pip install streamlit
pip install numpy
pip install librosa
pip install torch
pip install sounddevice
pip install wavio
pip install transformers
```

Alternatively, if a requirements.txt file is provided:

```bash
pip install -r requirements.txt
```

### Step 5: Install Additional System Dependencies

For audio processing, you may need to install system-level dependencies:

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install libsndfile1
sudo apt-get install portaudio19-dev
```

On macOS:

```bash
brew install libsndfile
brew install portaudio
```

## Usage

### Running the Application

Start the Streamlit application:

```bash
streamlit run main.py
```

The application will open in your default web browser, typically at `http://localhost:8501`

### Using the Application

#### Method 1: Upload Audio Files

1. Click on the "Browse files" button in the upload section
2. Select one or more audio files (WAV, MP3, FLAC, or OGG format)
3. The application will automatically process each file and display emotion predictions

#### Method 2: Record Audio

1. Click the "Record 4s Clip" button
2. Speak into your microphone when prompted
3. Wait for the recording to complete
4. The application will automatically analyze the recorded audio

### Understanding the Results

For each audio file, the application displays:

- Predicted emotion with an icon indicator
- Confidence percentage
- Probability distribution across all eight emotion classes
- Visual progress bars for each emotion probability

The summary section shows:

- Table of all predictions with files and confidence scores
- Bar chart showing emotion distribution across all processed files

## Model Information

- Model ID: `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3`
- Base Model: OpenAI Whisper Large V3
- Task: Audio Classification
- Emotions Detected: angry, calm, disgust, fear, happy, neutral, sad, surprise

## Configuration

### Audio Recording Settings

Default recording parameters in the code:

- Duration: 4 seconds
- Sample Rate: 22050 Hz
- Channels: 1 (mono)

### Audio Processing Settings

- Maximum audio duration: 30 seconds
- Sampling rate: Determined by the feature extractor
- Audio is automatically truncated or padded to fit model requirements

## Technical Details

### Model Loading

The model is loaded using Streamlit's caching mechanism to prevent redundant loading:

```python
@st.cache_resource
def load_hf_model()
```

### Audio Preprocessing

Audio files are:

1. Loaded using librosa
2. Converted to mono
3. Resampled if necessary
4. Truncated or padded to maximum length
5. Processed through the feature extractor

### Device Selection

The application automatically detects and uses:

- CUDA GPU if available
- CPU as fallback

## Troubleshooting

### Common Issues

#### Microphone Not Working

Ensure your browser has microphone permissions enabled for the application.

#### Model Download Fails

Check your internet connection. The model is downloaded automatically on first run from Hugging Face Hub.

#### Audio Format Not Supported

Convert your audio file to one of the supported formats: WAV, MP3, FLAC, or OGG.

#### Out of Memory Error

If processing large files or multiple files simultaneously, try:

- Processing fewer files at once
- Using shorter audio clips
- Ensuring sufficient RAM is available

### Performance Optimization

For faster inference:

- Use a CUDA-compatible GPU
- Process audio files in smaller batches
- Reduce the maximum audio duration if needed

## Project Structure

```
.
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies (if created)
└── README.md              # This file
```

## Dependencies

- streamlit: Web application framework
- numpy: Numerical computing
- librosa: Audio processing
- torch: Deep learning framework
- sounddevice: Audio recording
- wavio: WAV file writing
- transformers: Hugging Face model integration

## Notes

- The model is cached after first load to improve performance
- Audio files are temporarily stored during processing
- The application supports batch processing of multiple files
- All predictions include confidence scores and probability distributions

## Stopping the Application

To stop the Streamlit server:

- Press `Ctrl + C` in the terminal where the application is running

To deactivate the virtual environment:

```bash
deactivate
```
