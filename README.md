# 🛍️ Retail Product Discovery & Recommendation Chatbot

A multimodal AI-powered chatbot that provides intelligent product recommendations using image embeddings, text queries, and optional voice input. Built with Streamlit, PyTorch, CLIP, and FAISS for semantic similarity search.

---

## 📋 Project Overview

This project implements a **multimodal retail chatbot** that:
- Accepts **product images** for visual search
- Supports **text-based queries** for product recommendations
- Integrates **voice input** 
- Uses **CLIP embeddings** for semantic understanding
- Leverages **FAISS** for fast similarity search across 44,000+ products
- Displays **top-5 similar product recommendations**

---

## 🏗️ Project Structure

```
MultiModal Fusion Chatbot/
│
├── App.py                      # Main Streamlit application
├── data.csv                    # Product metadata (image names, descriptions)
├── data/                       # Directory containing 44,441 product images
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── ...
│   └── 44441.jpg
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
└── venv/                       # Virtual environment (after setup)
    ├── Scripts/
    ├── Lib/
    └── pyvenv.cfg
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (tested with Python 3.8.10)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **~3 GB free disk space** (for PyTorch and CLIP models)

### Installation Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/varshithgoud2407/Multimodel-Chatbot.git
cd "Multimodel Fusion Chatbot"
```

#### 2. Create Virtual Environment
```bash
python -m venv venv
```

#### 3. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\venv\Scripts\Activate.bat
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

#### 4. Upgrade pip
```bash
python -m pip install --upgrade pip
```

#### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

### Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| **streamlit** | 1.40.1 | Web UI framework |
| **torch** | 2.4.1 | Deep learning framework |
| **torchvision** | 0.19.1 | Computer vision utilities |
| **clip** | 1.0.1 | CLIP model for embeddings |
| **faiss-cpu** | 1.8.0 | Fast similarity search |
| **numpy** | 1.24.4 | Numerical computing |
| **pandas** | 2.0.3 | Data manipulation |
| **pillow** | 10.4.0 | Image processing |

### Optional
- **openai-whisper** (optional) - For voice transcription support

---

## 🎯 How to Run

### Start the Streamlit App
```bash
streamlit run App.py
```

The app will start and display:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Open in Browser
Navigate to: **http://localhost:8501**

---

## 💡 Usage Guide

### 1. **Image-Based Search**
- Click on **"📷 Upload Product Image"** in the sidebar
- Upload a JPG, PNG, or JPEG image
- The app will:
  - Generate CLIP embeddings for the image
  - Search FAISS index for similar products
  - Display top-5 recommendations

### 2. **Text-Based Query** (UI Prototype)
- Enter text in **"📝 Enter Text Query"** field
- Example: "Show me similar shoes"
- Currently displays a placeholder message (can be extended)

### 3. **Voice Input** (Optional)
- Upload an audio file (WAV/MP3) in **"🎤 Upload Voice Query"**
- If Whisper is installed, speech is transcribed to text
- Falls back gracefully if Whisper is unavailable

---

## 🔧 Technical Details

### Architecture

#### 1. **CLIP Model Loading**
- Loads pre-trained ViT-B/32 CLIP model
- Automatically selects **GPU** (if available) or **CPU**
- Cached on first run for faster subsequent loads

#### 2. **Image Indexing**
- Reads product images from `data/` folder
- Processes 44,441 images with progress tracking
- Generates CLIP embeddings (512-dimensional vectors)
- Normalizes embeddings using L2 normalization
- Builds FAISS IndexFlatIP for rapid similarity search

#### 3. **Recommendation Pipeline**
```
User Input Image → CLIP Embedding → L2 Normalization → FAISS Search → Top-5 Results
```

#### 4. **Performance Optimization**
- **Caching**: Uses Streamlit `@st.cache_resource` for model and index
- **Batch Processing**: FAISS handles efficient similarity search
- **L2 Normalization**: Ensures fair cosine similarity scoring

---

## 📊 Dataset Information

- **Total Products**: 44,441
- **Image Format**: JPG/PNG/JPEG
- **Metadata File**: `data.csv` (contains image filenames and descriptions)
- **Embedding Dimension**: 512 (CLIP ViT-B/32)

---

## ⚙️ Configuration

Edit these paths in `App.py` if your file structure differs:

```python
DATA_DIR = "."              # Project root directory
IMG_PATH = "data"           # Image folder path
CSV_PATH = "data.csv"       # Metadata CSV path
```

## 👤 Author

**Varshith Goud**

GitHub: [varshithgoud2407](https://github.com/varshithgoud2407)

---

## 🎓 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Official Guide](https://pytorch.org/tutorials/)
- [CLIP Paper](https://arxiv.org/abs/2103.14030)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

**Last Updated**: February 5, 2026  

