# app.py -- Streamlit demo for rPPG-only deepfake detection
# Single-file app: extracts rPPG -> features -> loads sklearn model -> predicts

import streamlit as st
import tempfile, os, shutil, glob
from pathlib import Path
import numpy as np
import cv2
from scipy import signal
import joblib

st.set_page_config(page_title="rPPG Deepfake Demo", layout="centered")

# -----------------------
# Config
# -----------------------
MODEL_PATH = "rppg_rf_model.joblib"   # put the trained model here or change path
DEFAULT_THRESHOLD = 0.5

# -----------------------
# Utility: face detection + rPPG extraction (minimal, robust)
# -----------------------
# Prefer MediaPipe if installed; fallback to Haar cascade.
USE_MEDIAPIPE = False
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    USE_MEDIAPIPE = True
    st.info("MediaPipe loaded — using it for face detection (faster/more robust).")
except Exception:
    st.warning("MediaPipe not available — falling back to OpenCV Haar cascade (works, but less robust).")

_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_bbox(frame):
    h, w = frame.shape[:2]
    if USE_MEDIAPIPE:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(img_rgb)
        if results.detections:
            det = results.detections[0].location_data.relative_bounding_box
            x = int(det.xmin * w); y = int(det.ymin * h)
            bw = int(det.width * w); bh = int(det.height * h)
            x = max(0, x); y = max(0, y)
            bw = min(w-x, max(10, bw)); bh = min(h-y, max(10, bh))
            return (x, y, bw, bh)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _haar.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        return faces[0]
    # center fallback
    s = min(h, w)
    return (int((w-s)/2), int((h-s)/2), s, s)

def extract_roi_mean_colors(video_path, max_frames=None, resize_width=480):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video: " + str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        fps = float(fps)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
    except Exception:
        fps = 30.0
    rgb_means = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if max_frames and frame_count > max_frames:
            break
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        bbox = detect_face_bbox(frame)
        if bbox is None:
            continue
        x, y, bw, bh = bbox
        roi = frame[y:y+bh, x:x+bw]
        if roi.size == 0:
            continue
        mean_rgb = np.mean(roi.reshape(-1, 3), axis=0)  # BGR
        mean_rgb = mean_rgb[::-1]  # to RGB
        rgb_means.append(mean_rgb)
    cap.release()
    if len(rgb_means) == 0:
        return None, fps
    return np.array(rgb_means), fps

def chrom_method(rgb_ts, fs=30.0):
    X = rgb_ts.astype(float)
    mean = X.mean(axis=0)
    Xn = X / (mean + 1e-8)
    M = np.array([[0, -2],
                  [1,  1],
                  [-1, 1]], dtype=float)
    S = np.dot(Xn, M)
    s1 = S[:,0]; s2 = S[:,1]
    std0 = s1.std(); std1 = s2.std()
    alpha = 0.0 if std1 == 0 else std0 / (std1 + 1e-8)
    chrom = s1 - alpha * s2
    chrom = signal.detrend(chrom)
    low = max(0.7 / fs, 1e-4)
    high = min(3.5 / fs, 0.999)
    try:
        if len(chrom) >= 6:
            b, a = signal.butter(3, [low, high], btype='band')
            chrom = signal.filtfilt(b, a, chrom)
    except Exception:
        chrom = chrom - np.mean(chrom)
    return chrom

def features_from_rppg(sig, fs=30.0):
    sig = sig - np.mean(sig)
    f, Pxx = signal.welch(sig, fs=fs, nperseg=min(256, max(64, len(sig))))
    hr_band = (f >= 0.7) & (f <= 3.5)
    if hr_band.sum() == 0:
        peak_bpm = 0.0; entropy = 0.0; peak_power = 0.0
    else:
        idx = np.argmax(Pxx[hr_band])
        peak_freq = f[hr_band][idx]
        peak_bpm = float(peak_freq * 60.0)
        Pnorm = Pxx[hr_band] / (np.sum(Pxx[hr_band]) + 1e-12)
        entropy = float(-np.sum(Pnorm * np.log(Pnorm + 1e-12)))
        peak_power = float(np.max(Pxx[hr_band]))
    std = float(np.std(sig))
    acorr = np.correlate(sig, sig, mode='full')
    acorr = acorr[acorr.size//2:]
    periodicity = float(np.max(acorr) / (acorr[0] + 1e-12)) if acorr.size>0 else 0.0
    return {'peak_bpm': peak_bpm, 'spec_entropy': entropy,
            'peak_power': peak_power, 'std': std, 'periodicity': periodicity}

# -----------------------
# Model loading
# -----------------------
MODEL = None
if os.path.exists(MODEL_PATH):
    try:
        MODEL = joblib.load(MODEL_PATH)
        st.success("Loaded model: " + MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model {MODEL_PATH}: {e}")
else:
    st.warning(f"Model not found at {MODEL_PATH}. Put your trained joblib model there.")

def compute_features_from_video(video_path):
    rgb, fps = extract_roi_mean_colors(video_path, max_frames=800, resize_width=480)
    if rgb is None:
        raise ValueError("No face/ROI found in video.")
    sig = chrom_method(rgb, fs=fps if fps>0 else 30.0)
    feats = features_from_rppg(sig, fs=fps if fps>0 else 30.0)
    order = ['peak_bpm','spec_entropy','peak_power','std','periodicity']
    X = np.array([feats[k] for k in order], dtype=float).reshape(1, -1)
    return X, feats, int(fps), int(rgb.shape[0])

def predict_video_rppg(video_path, model=MODEL, threshold=DEFAULT_THRESHOLD):
    if model is None:
        raise ValueError("No model available. Train and save model to 'rppg_rf_model.joblib'")
    X, feats, fps, frames = compute_features_from_video(video_path)
    if hasattr(model, 'predict_proba'):
        prob = float(model.predict_proba(X)[0,1])
    else:
        pred = int(model.predict(X)[0])
        prob = float(pred)
    label = int(prob >= threshold)
    return {'prob_fake': prob, 'label': label, 'features': feats, 'fps': fps, 'frames': frames}

# -----------------------
# Streamlit UI
# -----------------------
st.title("rPPG Deepfake Detection — demo")
st.markdown("Upload a short face video (<=10s). The app extracts physiological signal (rPPG) and uses a saved model to predict fake vs real.")

uploaded = st.file_uploader("Upload a short video", type=['mp4','mov','avi','mkv'])
threshold = st.slider("Decision threshold (probability of fake)", 0.0, 1.0, float(DEFAULT_THRESHOLD), 0.01)

if uploaded is not None:
    tmpdir = tempfile.mkdtemp()
    try:
        tmpfile = os.path.join(tmpdir, uploaded.name)
        with open(tmpfile, 'wb') as f:
            f.write(uploaded.read())
        st.info(f"Saved uploaded file to {tmpfile}")

        # run prediction
        try:
            res = predict_video_rppg(tmpfile, model=MODEL, threshold=threshold)
            st.metric("Probability of fake", f"{res['prob_fake']:.3f}")
            st.write("Prediction:", "DEEPFAKE" if res['label']==1 else "REAL")
            st.write("rPPG features:", res['features'])
            st.write("Video frames:", res['frames'], "FPS:", res['fps'])
            # optional: show a waveform preview
            rgb, fps = extract_roi_mean_colors(tmpfile, max_frames=800, resize_width=480)
            sig = chrom_method(rgb, fs=fps if fps>0 else 30.0)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1,1,figsize=(8,2))
            ax.plot(sig); ax.set_title('rPPG trace (sampled ROI mean)'); ax.set_xlabel('frame')
            st.pyplot(fig)
        except Exception as e:
            st.error("Error during prediction: " + str(e))
    finally:
        shutil.rmtree(tmpdir)
else:
    st.info("No file uploaded yet. You can test using one of your sample videos.")

st.markdown("---")
st.write("Notes: rPPG is sensitive to lighting/motion/compression. For production, ensemble with a pretrained visual model.")
