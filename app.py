import os
import cv2
import tempfile
import urllib.request
import numpy as np
from PIL import Image
import librosa
import streamlit as st

import torch
import torch.nn as nn
import torchvision.models as models
from facenet_pytorch import MTCNN

# ----------------- Configuration & Paths -----------------
# Replace this URL with your exact GitHub Release download link
WEIGHTS_URL = "https://github.com/Raghuvanshiii/Multimodal-deepfake-detector/releases/download/v1.0/best_deepfake_detector.pt"
LOCAL_WEIGHTS_PATH = "best_deepfake_detector.pt"

device = torch.device("cpu")

st.set_page_config(
    page_title="Multimodal Deepfake Detector",
    page_icon="🛡️",
    layout="centered"
)

# ----------------- Model Architecture -----------------
class AudioCNN(nn.Module):
    """Extracts high-level spectral features from a Log-Mel Spectrogram."""
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
    """Unified framework fusing spatial ResNet embeddings, audio CNN, and temporal self-attention."""
    def __init__(self, embed_dim=512, nhead=4, num_transformer_layers=2):
        super(MultimodalDeepfakeDetector, self).__init__()
        
        # Spatial backbone
        resnet = models.resnet18(weights=None)
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_proj = nn.Linear(512, embed_dim)
        
        # Audio backbone
        self.audio_backbone = AudioCNN(embed_dim=embed_dim)
        
        # Temporal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.2,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Multimodal fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, faces, spectrogram):
        # faces: [B, T, C, H, W]
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
    # 1. Download weights from GitHub Release if not present locally
    if not os.path.exists(LOCAL_WEIGHTS_PATH):
        with st.spinner("Downloading trained model weights from GitHub Release..."):
            urllib.request.urlretrieve(WEIGHTS_URL, LOCAL_WEIGHTS_PATH)

    # 2. Instantiate and load weights onto CPU
    detector_model = MultimodalDeepfakeDetector().to(device)
    raw_weights = torch.load(LOCAL_WEIGHTS_PATH, map_location=device, weights_only=True)
    cleaned_weights = {k.replace("module.", ""): v for k, v in raw_weights.items()}
    detector_model.load_state_dict(cleaned_weights)
    detector_model.eval()

    # 3. Initialize MTCNN face detector
    face_detector = MTCNN(
        image_size=224,
        margin=20,
        keep_all=False,
        select_largest=True,
        post_process=True,
        device=device
    )
    return detector_model, face_detector

model, mtcnn = load_models()

# ----------------- Web UI -----------------
st.title("🛡️ Multimodal Deepfake Detection System")
st.markdown(
    """
    This system examines audio-visual artifacts across video streams by coupling:
    - **Spatial facial extraction** (MTCNN + ResNet-18)
    - **Temporal sequence analysis** (Multi-Head Self-Attention Transformer)
    - **Acoustic synchronization** (Mel-Spectrogram 2D CNN)
    """
)

uploaded_file = st.file_uploader("Upload an MP4 Video to Analyze", type=["mp4"])

if uploaded_file is not None:
    # Save uploaded file temporarily for OpenCV and Librosa processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    st.video(video_path)

    with st.spinner("Analyzing spatial-temporal facial coherence and audio track..."):
        # --- 1. Visual Stream Extraction ---
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(total_frames - 1, 0), 16, dtype=int)
        face_tensors = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                face_tensors.append(torch.zeros(3, 224, 224))
                continue
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                face = mtcnn(pil_img)
                face_tensors.append(face if face is not None else torch.zeros(3, 224, 224))
            except Exception:
                face_tensors.append(torch.zeros(3, 224, 224))
        cap.release()
        faces_tensor = torch.stack(face_tensors[:16]).unsqueeze(0).to(device)

        # --- 2. Audio Stream Extraction ---
        try:
            y, _ = librosa.load(video_path, sr=16000)
            if len(y) < 160000:
                y = np.pad(y, (0, 160000 - len(y)))
            else:
                y = y[:160000]
            mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128, n_fft=2048, hop_length=512)
            log_mel = librosa.power_to_db(mel, ref=np.max)
            log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
            audio_tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        except Exception:
            audio_tensor = torch.zeros(1, 1, 128, 313).to(device)

        # --- 3. Forward Pass ---
        with torch.no_grad():
            logit = model(faces_tensor, audio_tensor)
            fake_prob = float(torch.sigmoid(logit).item())

        real_prob = 1.0 - fake_prob

    # --- 4. Render Verdict ---
    st.divider()
    if fake_prob >= 0.5:
        st.error(f"### Verdict: DEEPFAKE (Manipulated)")
        st.markdown(f"**Confidence Score:** `{fake_prob * 100:.2f}%`")
    else:
        st.success(f"### Verdict: AUTHENTIC (Real)")
        st.markdown(f"**Confidence Score:** `{real_prob * 100:.2f}%`")

    st.progress(fake_prob, text=f"Deepfake Probability: {fake_prob * 100:.1f}%")
