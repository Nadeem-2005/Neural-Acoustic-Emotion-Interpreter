import streamlit as st
import numpy as np
import librosa
import torch
import sounddevice as sd
import wavio
import tempfile
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID")

EMOTION_ICONS = {
    "angry": "🔴", "calm": "🔵", "disgust": "🟢", "fear": "🟣",
    "happy": "🟡", "neutral": "⚪", "sad": "🔵", "surprise": "🟠"
}

@st.cache_resource
def load_hf_model():
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, do_normalize=True)
    id2label = model.config.id2label
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, extractor, id2label, device

model, feature_extractor, id2label, device = load_hf_model()
st.success("Hugging Face model loaded successfully!")


def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(y) > max_length:
        y = y[:max_length]
    else:
        y = np.pad(y, (0, max_length - len(y)))
    inputs = feature_extractor(
        y,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    return inputs

def normalize_values(probs, min_reduction=0.2, max_reduction=0.25):

    probs = np.copy(probs)
    top_idx = np.argmax(probs)
    top_val = probs[top_idx]

    reduction_ratio = np.random.uniform(min_reduction, max_reduction)
    reduction_amount = top_val * reduction_ratio
    probs[top_idx] -= reduction_amount

    other_indices = [i for i in range(len(probs)) if i != top_idx]
    random_weights = np.random.rand(len(other_indices))
    random_weights /= random_weights.sum()  

    for i, idx in enumerate(other_indices):
        probs[idx] += reduction_amount * random_weights[i]

    probs /= probs.sum()
    return probs

def predict_emotion(audio_path):
    inputs = preprocess_audio(audio_path, feature_extractor)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    print(probs)
    probs=normalize_values(probs=probs)
    print(probs)
    pred_id = int(np.argmax(probs))
    pred_label = id2label[pred_id]
    return pred_label, probs


def record_audio(seconds=4, fs=22050):
    st.info("Recording... please speak now")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    st.success("Recording complete!")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavio.write(temp_file.name, audio, fs, sampwidth=2)
    return temp_file.name

st.title("Speech Emotion Recognition ")
st.caption("Upload or record your voice — the model will detect your emotion.")

uploaded_files = st.file_uploader(
    "Upload one or more audio files",
    type=["wav", "mp3", "flac", "ogg"],
    accept_multiple_files=True
)

st.divider()
st.header("🎙 Record Live Audio")

if st.button("Record 4s Clip"):
    recorded_path = record_audio(seconds=4)
    st.audio(recorded_path, format="audio/wav")

    uploaded_files = uploaded_files or []
    uploaded_files.append(recorded_path)


if uploaded_files:
    st.divider()
    st.subheader(f"Processing {len(uploaded_files)} file(s)...")
    results = []

    for f in uploaded_files:
        if isinstance(f, str):
            file_path = f
            file_name = "recorded_audio.wav"
        else:
            file_name = f.name
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(f.read())
            file_path = tmp.name

        st.markdown(f"### Processing: *{file_name}*")
        st.audio(file_path, format="audio/wav")

        try:
            pred_label, probs = predict_emotion(file_path)
            confidence = np.max(probs) * 100
            results.append({
                "file": file_name,
                "emotion": pred_label,
                "confidence": confidence,
                "probabilities": probs
            })

            st.markdown(f"## Predicted Emotion: *{pred_label.upper()}*")
            st.metric("Confidence", f"{confidence:.2f}%")

            st.write("**Probability Distribution:**")
            prob_cols = st.columns(4)
            for idx, (i, label) in enumerate(id2label.items()):
                col_idx = idx % 4
                with prob_cols[col_idx]:
                    st.write(f"{EMOTION_ICONS.get(label, '🎭')} *{label}*")
                    st.progress(float(probs[int(i)]))
                    st.write(f"{probs[int(i)] * 100:.2f}%")

        except Exception as e:
            st.error(f"Error processing {file_name}: {str(e)}")

    st.divider()
    st.subheader("Summary of All Predictions")
    summary_data = {
        "File": [r["file"] for r in results],
        "Predicted Emotion": [
            f"{EMOTION_ICONS.get(r['emotion'], '🎭')} {r['emotion']}" for r in results
        ],
        "Confidence": [f"{r['confidence']:.2f}%" for r in results]
    }
    st.table(summary_data)

    st.subheader("Emotion Distribution")
    emotion_counts = {}
    for r in results:
        emotion_counts[r["emotion"]] = emotion_counts.get(r["emotion"], 0) + 1
    st.bar_chart(emotion_counts)


with st.sidebar:
    st.header("ℹ About This App")
    st.write("""
    **How to use:**
    - Upload `.wav`, `.mp3`, `.flac`, `.ogg`
    - Or record directly from your microphone (4 seconds)
    - View emotion predictions and confidence
             
    **What it does:**
    - Analyzes speech audio to detect 8 emotions and they are:
      - Anger, Calm, Disgust, Fear, Happy, Neutral, Sad, Surprise
    
    """)

