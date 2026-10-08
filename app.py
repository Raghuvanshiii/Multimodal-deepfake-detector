import os
import cv2
import tempfile
import urllib.request
import numpy as np
from PIL import Image
import librosa
import matplotlib.pyplot as plt
import streamlit as st

import torch
import torch.nn as nn
import torchvision.models as models
from facenet_pytorch import MTCNN

# ----------------- Configuration & Page Setup -----------------
st.set_page_config(
    page_title="Multimodal Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Replace with your actual GitHub Release download link
WEIGHTS_URL = "https://github.com/<your-username>/<your-repo>/releases/download/v1.0/best_deepfake_detector.pt"
LOCAL_WEIGHTS_PATH = "best_deepfake_detector.pt"
device = torch.device("cpu")

# ----------------- Dynamic Theme CSS Injection -----------------
with st.sidebar:
    st.header("⚙️ Display & Settings")
    theme_choice = st.radio("App Theme", ["System Default", "Dark Mode", "Light Mode"], index=0)
    st.divider()
    threshold = st.slider("Classification Threshold (Fake %)", 10, 90, 50, step=5) / 100.0

# Apply custom theme variables according to the user selection
if theme_choice == "Dark Mode":
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #ffffff; }
        .metric-card { background: #1f2937; border-radius: 12px; padding: 18px; border: 1px solid #374151; }
        </style>
    """, unsafe_allow_html=True)
elif theme_choice == "Light Mode":
    st.markdown("""
        <style>
        .stApp { background-color: #f9fafb; color: #111827; }
        .metric-card { background: #ffffff; border-radius: 12px; padding: 18px; border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .metric-card { border-radius: 12px; padding: 18px; border: 1px solid rgba(128,128,128,0.2); }
        </style>
    """, unsafe_allow_html=True)

# ----------------- Model Architecture -----------------
class AudioCNN(nn.Module):
    def __init__(self, embed_dim=512):
        super(AudioCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        feat = self.conv(x)
        return self.fc(feat.view(feat.size(0), -1))

class MultimodalDeepfakeDetector(nn.Module):
    def __init__(self, embed_dim=512, nhead=4, num_transformer_layers=2):
        super(MultimodalDeepfakeDetector, self).__init__()
        resnet = models.resnet18(weights=None)
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_proj = nn.Linear(512, embed_dim)
        self.audio_backbone = AudioCNN(embed_dim=embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=1024, dropout=0.2, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, faces, spectrogram):
        b, t, c, h, w = faces.shape
        faces_reshaped = faces.view(b * t, c, h, w)
        vis = self.visual_backbone(faces_reshaped).view(b * t, -1)
        vis = self.visual_proj(vis).view(b, t, -1)
        
        temp_out = self.temporal_transformer(vis)
        video_repr = temp_out.mean(dim=1)
        audio_repr = self.audio_backbone(spectrogram)
        
        fused = torch.cat((video_repr, audio_repr), dim=1)
        return self.classifier(fused).squeeze(1)

# ----------------- Resource Initialization (Cached) -----------------
@st.cache_resource(show_spinner=False)
def load_models():
    if not os.path.exists(LOCAL_WEIGHTS_PATH):
        with st.spinner("Downloading trained model weights from GitHub Release..."):
            urllib.request.urlretrieve(WEIGHTS_URL, LOCAL_WEIGHTS_PATH)

    detector_model = MultimodalDeepfakeDetector().to(device)
    raw_weights = torch.load(LOCAL_WEIGHTS_PATH, map_location=device, weights_only=True)
    cleaned_weights = {k.replace("module.", ""): v for k, v in raw_weights.items()}
    detector_model.load_state_dict(cleaned_weights)
    detector_model.eval()

    face_detector = MTCNN(
        image_size=224, margin=20, keep_all=False, select_largest=True, post_process=True, device=device
    )
    return detector_model, face_detector

model, mtcnn = load_models()

# Sidebar Metadata Cards
with st.sidebar:
    st.subheader("Model Specifications")
    st.markdown("- **Visual Stream:** MTCNN + ResNet-18")
    st.markdown("- **Temporal Stream:** 2-Layer Transformer")
    st.markdown("- **Audio Stream:** 128-Mel Spectrogram CNN")
    st.markdown("- **Fusion Type:** Concatenation Bottleneck")

# ----------------- Header Banner -----------------
st.title("🛡️ Multimodal Deepfake Forensic Analyzer")
st.caption("Cross-modal temporal synthesis verification using ResNet spatial embeddings and Mel-frequency acoustic alignment.")

# ----------------- File Input & Two-Column Layout -----------------
uploaded_file = st.file_uploader("Upload MP4 Video for Forensic Evaluation", type=["mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.subheader("📹 Input Stream Preview")
        st.video(video_path)

    with col_right:
        st.subheader("🔍 Prediction & Forensic Analysis")
        analyze_btn = st.button("🚀 Analyze Video Stream", use_container_width=True, type="primary")

    if analyze_btn:
        progress_bar = st.progress(0, text="Initializing multimodal extraction...")
        
        # --- Step 1: Visual Extraction ---
        progress_bar.progress(20, text="Detecting & aligning facial frames (MTCNN)...")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(total_frames - 1, 0), 16, dtype=int)
        
        face_tensors = []
        preview_faces = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                face_tensors.append(torch.zeros(3, 224, 224))
                continue
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                face = mtcnn(pil_img)
                if face is not None:
                    face_tensors.append(face)
                    # Denormalize tensor for visual preview
                    preview_img = (face.permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
                    preview_faces.append(preview_img)
                else:
                    face_tensors.append(torch.zeros(3, 224, 224))
            except Exception:
                face_tensors.append(torch.zeros(3, 224, 224))
        cap.release()
        faces_tensor = torch.stack(face_tensors[:16]).unsqueeze(0).to(device)

        # --- Step 2: Audio Extraction ---
        progress_bar.progress(60, text="Generating acoustic Log-Mel spectrogram...")
        log_mel_data = None
        try:
            y, _ = librosa.load(video_path, sr=16000)
            y = np.pad(y, (0, 160000 - len(y))) if len(y) < 160000 else y[:160000]
            mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128, n_fft=2048, hop_length=512)
            log_mel = librosa.power_to_db(mel, ref=np.max)
            log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
            log_mel_data = log_mel
            audio_tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        except Exception:
            audio_tensor = torch.zeros(1, 1, 128, 313).to(device)

        # --- Step 3: Model Inference ---
        progress_bar.progress(90, text="Executing Transformer cross-modal attention...")
        with torch.no_grad():
            logit = model(faces_tensor, audio_tensor)
            fake_prob = float(torch.sigmoid(logit).item())

        real_prob = 1.0 - fake_prob
        progress_bar.progress(100, text="Analysis complete!")

        # --- Step 4: Verdict & Confidence Cards ---
        with col_right:
            is_fake = fake_prob >= threshold
            
            if is_fake:
                st.error("### ⚠️ Verdict: DEEPFAKE (Manipulated)")
            else:
                st.success("### ✅ Verdict: AUTHENTIC (Real)")

            # Metric Cards
            m1, m2 = st.columns(2)
            m1.metric("Deepfake Probability", f"{fake_prob * 100:.2f}%")
            m2.metric("Authentic Confidence", f"{real_prob * 100:.2f}%")

            # Dual Distribution Bars
            st.write("**Confidence Breakdown:**")
            st.progress(fake_prob, text=f"Deepfake Risk: {fake_prob * 100:.1f}%")
            st.progress(real_prob, text=f"Authenticity Score: {real_prob * 100:.1f}%")

        # --- Step 5: Visual Explainability Tabs ---
        st.divider()
        tab1, tab2 = st.tabs(["🖼️ Extracted Facial Frames (MTCNN)", "🎵 Acoustic Spectrogram (Librosa)"])

        with tab1:
            st.write("Aligned facial frames sampled uniformly across the video sequence:")
            if len(preview_faces) > 0:
                cols = st.columns(min(len(preview_faces), 8))
                for i, img in enumerate(preview_faces[:8]):
                    cols[i].image(img, caption=f"Frame {i+1}", use_container_width=True)
            else:
                st.info("No prominent faces were detected in the sampled frames.")

        with tab2:
            st.write("Normalized Log-Mel Spectrogram representing speech frequency bins:")
            if log_mel_data is not None:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.imshow(log_mel_data, aspect='auto', origin='lower', cmap='magma')
                ax.set_title("Log-Mel Spectrogram (16 kHz, 128 Bins)")
                ax.set_xlabel("Time Frame Bins")
                ax.set_ylabel("Mel Frequency Bins")
                st.pyplot(fig)
            else:
                st.info("No valid audio track extracted from the video container.")
