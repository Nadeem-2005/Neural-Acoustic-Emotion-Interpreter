import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
import io
import sounddevice as sd
from datetime import datetime
import time

# --- EMOTION MAPPING ---
EMOTION_LABELS = {
    0: 'angry',
    1: 'calm',
    2: 'disgust',
    3: 'fear',
    4: 'happy',
    5: 'neutral',
    6: 'sad',
    7: 'surprise'
}

# Emotion colors for visual feedback
EMOTION_COLORS = {
    'angry': '🔴',
    'calm': '🔵',
    'disgust': '🟢',
    'fear': '🟣',
    'happy': '🟡',
    'neutral': '⚪',
    'sad': '🔵',
    'surprise': '🟠'
}

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_cnn_model():
    model = load_model("saved_models/cnn_model.h5")
    return model

model = load_cnn_model()
st.success("✅ Model loaded successfully!")

# --- 2. FEATURE EXTRACTION FUNCTION ---
def extract_mfcc_features(audio_data, sr, n_mfcc=40, target_length=162):
    """Extract MFCC features and reshape to match Conv1D input (batch, time_steps, 1)"""
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)  # shape (n_mfcc, time)
    mfcc = mfcc.T  # shape (time, n_mfcc)

    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

    # Pad or truncate
    if mfcc.shape[0] < target_length:
        mfcc = np.pad(mfcc, ((0, target_length - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:target_length, :]

    # ✅ FIX: Collapse 40 MFCCs → 1 feature (mean across coefficients)
    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)  # shape (time, 1)

    # Add batch dimension → (1, time_steps, 1)
    mfcc_final = np.expand_dims(mfcc_mean, axis=0)
    return mfcc_final
# --- 3. AUDIO RECORDING FUNCTION ---
def record_audio(duration, sample_rate):
    """Record audio with proper blocking"""
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        blocking=True
    )
    return recording.flatten()

# --- 4. AUDIO RECORDING SECTION ---
st.header("🎙️ Record Live Audio")

col1, col2 = st.columns(2)

with col1:
    duration = st.slider("Recording duration (seconds)", 1, 10, 3)
    sample_rate = st.selectbox("Sample rate (Hz)", [16000, 22050, 44100], index=1)

with col2:
    st.write("Click 'Record' to start recording")
    record_button = st.button("🔴 Record Audio", type="primary")

if record_button:
    countdown_placeholder = st.empty()
    status_placeholder = st.empty()
    
    for i in range(3, 0, -1):
        countdown_placeholder.markdown(f"## Recording starts in {i}...")
        time.sleep(1)
    
    countdown_placeholder.markdown("## 🔴 RECORDING NOW!")
    progress_bar = st.progress(0)
    
    try:
        import threading
        audio_data = None
        recording_complete = False
        
        def do_recording():
            global audio_data, recording_complete
            audio_data = record_audio(duration, sample_rate)
            recording_complete = True
        
        record_thread = threading.Thread(target=do_recording)
        record_thread.start()
        
        start_time = time.time()
        while not recording_complete:
            elapsed = time.time() - start_time
            progress = min(elapsed / duration, 1.0)
            progress_bar.progress(progress)
            time.sleep(0.1)
        
        record_thread.join()
        progress_bar.progress(1.0)
        
        countdown_placeholder.empty()
        status_placeholder.success("✅ Recording completed!")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.audio(audio_data, sample_rate=sample_rate)
        
        # Process recorded audio
        st.write("🔄 Processing recorded audio...")
        mfcc_features = extract_mfcc_features(audio_data, sample_rate)
        st.write(f"Feature shape: {mfcc_features.shape}")
        
        # ✅ Predict
        predictions = model.predict(mfcc_features, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_emotion = EMOTION_LABELS[predicted_class]
        confidence = predictions[0][predicted_class] * 100
        
        # Display result
        st.markdown(f"### {EMOTION_COLORS[predicted_emotion]} Predicted Emotion: **{predicted_emotion.upper()}**")
        st.metric("Confidence", f"{confidence:.2f}%")
        
        # Show probability distribution
        st.write("**Probability Distribution:**")
        prob_cols = st.columns(4)
        for idx, (label_idx, prob) in enumerate(zip(range(len(predictions[0])), predictions[0])):
            col_idx = idx % 4
            emotion_name = EMOTION_LABELS[label_idx]
            with prob_cols[col_idx]:
                st.write(f"{EMOTION_COLORS[emotion_name]} **{emotion_name}**")
                st.progress(float(prob))
                st.write(f"{prob*100:.2f}%")
                
    except Exception as e:
        countdown_placeholder.empty()
        status_placeholder.error(f"❌ Error during recording: {str(e)}")
        st.error("Make sure your microphone is connected and permissions are granted.")

# --- 5. FILE UPLOAD SECTION ---
st.divider()
st.header("📤 Upload Audio Files")

uploaded_files = st.file_uploader(
    "Upload one or more audio files (.wav, .mp3, .npy)",
    type=["mp3", "wav", "npy"],
    accept_multiple_files=True
)

# --- 6. PROCESS UPLOADED FILES ---
if uploaded_files:
    st.write(f"📂 {len(uploaded_files)} file(s) uploaded.")
    results = []
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        st.divider()
        st.write(f"🎧 Processing: **{file_name}**")
        
        try:
            if file_name.endswith(".npy"):
                x_input = np.load(uploaded_file, allow_pickle=True)
                st.info("Loaded preprocessed .npy file")
            elif file_name.endswith((".wav", ".mp3")):
                y, sr = librosa.load(uploaded_file, sr=22050)
                st.write(f"Audio duration: {len(y)/sr:.2f} seconds")
                st.audio(uploaded_file)
                x_input = extract_mfcc_features(y, sr)
                st.success("✅ Features extracted successfully")
            else:
                st.error(f"❌ Unsupported file format: {file_name}")
                continue
            
            st.write(f"Input shape: {x_input.shape}")
            
            # ✅ Predict
            predictions = model.predict(x_input, verbose=0)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_emotion = EMOTION_LABELS[predicted_class]
            confidence = predictions[0][predicted_class] * 100
            
            results.append({
                "file": file_name,
                "emotion": predicted_emotion,
                "class": predicted_class,
                "confidence": confidence,
                "probabilities": predictions[0]
            })
            
            st.markdown(f"### {EMOTION_COLORS[predicted_emotion]} Prediction: **{predicted_emotion.upper()}**")
            st.metric("Confidence", f"{confidence:.2f}%")
            
            st.write("**Probability Distribution:**")
            prob_cols = st.columns(4)
            for idx, (label_idx, prob) in enumerate(zip(range(len(predictions[0])), predictions[0])):
                col_idx = idx % 4
                emotion_name = EMOTION_LABELS[label_idx]
                with prob_cols[col_idx]:
                    st.write(f"{EMOTION_COLORS[emotion_name]} **{emotion_name}**")
                    st.progress(float(prob))
                    st.write(f"{prob*100:.2f}%")
            
        except Exception as e:
            st.error(f"❌ Error processing {file_name}: {str(e)}")
            continue
    
    if results:
        st.divider()
        st.subheader("📊 Summary of All Predictions")
        summary_data = {
            "File": [r["file"] for r in results],
            "Predicted Emotion": [f"{EMOTION_COLORS[r['emotion']]} {r['emotion']}" for r in results],
            "Confidence": [f"{r['confidence']:.2f}%" for r in results]
        }
        st.table(summary_data)
        
        st.subheader("📈 Emotion Distribution")
        emotion_counts = {}
        for r in results:
            emotion = r['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        st.bar_chart(emotion_counts)

# --- 8. SIDEBAR INFO ---
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses a CNN model to predict emotions from audio.
    
    **Supported Emotions:**
    """)
    for idx, emotion in EMOTION_LABELS.items():
        st.write(f"{EMOTION_COLORS[emotion]} {idx}: {emotion}")
    
    st.divider()
    st.write("**Tips for better results:**")
    st.write("- Use clear audio without background noise")
    st.write("- Speak naturally and expressively")
    st.write("- Recording duration: 3-5 seconds works best")
    st.write("- Allow microphone permissions when prompted")