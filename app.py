import base64
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
import numpy as np
import os
import soundfile as sf
import librosa

from model import FADCNN
from pathlib import Path

# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="Fake Audio Detection Web App", version="2.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Audio utilities (NO torchaudio)
# -----------------------------
def load_audio(path, target_sr=16000):
    wav, sr = sf.read(path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)  # stereo → mono
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    wav = wav.astype("float32")
    wav = wav / (np.max(np.abs(wav)) + 1e-9)
    return wav, sr


def preprocess_audio(path, n_mels=64, target_frames=128):
    wav, sr = load_audio(path)

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    if mel.shape[1] < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])))
    else:
        mel = mel[:, :target_frames]

    return mel.astype("float32")

# -----------------------------
# Load CNN model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = Path("models/cnn_v2.pt")

model = FADCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("✅ CNN model loaded")

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        tmp_path = f"temp_{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        wav, sr = load_audio(tmp_path)
        import base64

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        mel = preprocess_audio(tmp_path)

        x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()

        label = "FAKE" if prob >= 0.5 else "REAL"

        os.remove(tmp_path)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": {
                    "filename": file.filename,
                    "label": label,
                    "prob": round(prob, 3),
                    "mel_data": mel.tolist(),
                    "waveform": wav[:5000].tolist(),  # limit size
                    "audio_base64": audio_base64,

                    "explanation": "Prediction successful"
                },
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e), "result": None},
        )
