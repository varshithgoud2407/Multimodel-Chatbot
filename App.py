import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
import faiss
import clip

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Retail Product Discovery & Recommendation Chatbot",
    layout="wide"
)

st.title("🛍️ Retail Product Discovery & Recommendation Chatbot")
st.write("Upload a product image, enter text, or optionally use voice input.")

# =========================
# PATHS (EDIT IF NEEDED)
# =========================
DATA_DIR = "."
IMG_PATH = "data"
CSV_PATH = "data.csv"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

df = load_data()

# =========================
# LOAD CLIP MODEL
# =========================
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device

clip_model, clip_preprocess, device = load_clip_model()

# =========================
# SAFE WHISPER LOADER
# =========================
def load_whisper_safe():
    try:
        import whisper
        return whisper.load_model("base")
    except Exception:
        return None

# =========================
# NUMPY NORMALIZATION
# =========================
def l2_normalize(x, axis=1, eps=1e-10):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)

# =========================
# CLIP IMAGE EMBEDDING
# =========================
def extract_clip_embedding(image):
    image = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

# =========================
# BUILD IMAGE INDEX (WITH % PROGRESS)
# =========================
@st.cache_resource
def build_image_index():
    image_embeddings = []
    image_paths = []

    total_images = len(df)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, img_name in enumerate(df["image"]):
        img_path = os.path.join(IMG_PATH, img_name)
        image = Image.open(img_path).convert("RGB")
        emb = extract_clip_embedding(image)

        image_embeddings.append(emb)
        image_paths.append(img_path)

        percent = int(((idx + 1) / total_images) * 100)

        # Streamlit UI
        progress_bar.progress(percent)
        status_text.text(
            f"Building image index: {percent}% ({idx+1}/{total_images})"
        )

        # CLI output
        sys.stdout.write(
            f"\rBuilding image index: {percent}% ({idx+1}/{total_images})"
        )
        sys.stdout.flush()

    print("\nImage indexing completed ✅")

    image_embeddings = np.vstack(image_embeddings)
    image_embeddings = l2_normalize(image_embeddings)

    index = faiss.IndexFlatIP(image_embeddings.shape[1])
    index.add(image_embeddings)

    return index, image_paths

index, image_paths = build_image_index()

# =========================
# RECOMMENDATION FUNCTION
# =========================
def recommend_similar_products(query_image, top_k=5):
    query_emb = extract_clip_embedding(query_image)
    query_emb = l2_normalize(query_emb)
    _, indices = index.search(query_emb, top_k)
    return [image_paths[i] for i in indices[0]]

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("🔧 Input Options")

uploaded_image = st.sidebar.file_uploader(
    "📷 Upload Product Image",
    type=["jpg", "png", "jpeg"]
)

uploaded_audio = st.sidebar.file_uploader(
    "🎤 Upload Voice Query (optional)",
    type=["wav", "mp3"]
)

text_query = st.sidebar.text_input(
    "📝 Enter Text Query",
    placeholder="e.g. Show me similar shoes"
)

# =========================
# HANDLE INPUT
# =========================
st.subheader("📥 User Input")

query_image = None

if uploaded_image:
    query_image = Image.open(uploaded_image).convert("RGB")
    st.image(query_image, caption="Uploaded Image", width=250)

elif uploaded_audio:
    st.audio(uploaded_audio)

    whisper_model = load_whisper_safe()
    if whisper_model is None:
        st.warning(
            "⚠️ Whisper not available. Voice transcription disabled, "
            "but the UI prototype works correctly."
        )
    else:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_audio.read())

        result = whisper_model.transcribe("temp_audio.wav")
        text_query = result["text"]
        st.write("🎤 Transcribed Text:", text_query)

elif text_query:
    st.write("📝 Text Query:", text_query)

else:
    st.info("Please upload an image, audio, or enter text to continue.")

# =========================
# SHOW RECOMMENDATIONS
# =========================
st.subheader("🔍 Recommended Products")

if query_image is not None:
    recommendations = recommend_similar_products(query_image, top_k=5)

    cols = st.columns(5)
    for col, img_path in zip(cols, recommendations):
        img = Image.open(img_path)
        col.image(img, use_column_width=True)

elif text_query:
    st.info("Text-based recommendations can be connected to the backend model.")

else:
    st.info("Recommendations will appear here after input.")
