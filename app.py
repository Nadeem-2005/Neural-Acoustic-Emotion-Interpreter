# ---------------------------------------------------------
# STREAMLIT APP FOR CNN MODEL DEPLOYMENT
# ---------------------------------------------------------
import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="CNN Model Deployment", layout="centered")
st.title("🧠 CNN Model - Streamlit Deployment")
st.write("Upload your data to get predictions using the trained CNN model.")

# --- 2. LOAD TRAINED MODEL ---
@st.cache_resource  # caches the model so it doesn't reload every time
def load_cnn_model():
    model = load_model("saved_models/cnn_model.h5")
    return model

model = load_cnn_model()
st.success("✅ Model loaded successfully!")

# --- 3. DATA INPUT SECTION ---
st.subheader("📤 Upload your test data (NumPy .npy file)")

uploaded_file = st.file_uploader("Upload test file", type=["npy"])
if uploaded_file is not None:
    x_input = np.load(uploaded_file)
    st.write("✅ Data uploaded successfully.")
    st.write("Input shape:", x_input.shape)

    # If input is 2D, reshape for CNN
    if len(x_input.shape) == 2:
        x_input = np.expand_dims(x_input, axis=2)
        st.info("Reshaped input for CNN: " + str(x_input.shape))

    # --- 4. MAKE PREDICTION ---
    preds = model.predict(x_input)
    predicted_classes = np.argmax(preds, axis=1)

    st.subheader("🔍 Predictions")
    st.write(predicted_classes)

    # Optionally show probabilities
    st.write("Prediction Probabilities:")
    st.write(preds)

else:
    st.warning("Please upload a .npy file to make predictions.")

# --- 5. FOOTER ---
st.markdown("---")
st.markdown("💻 Developed by **Nadeem** | CNN Deployment Demo")