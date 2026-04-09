#  Spatiotemporal Attention and Cross-Modal Fusion Framework for Intelligent Piano Learning with Personalized Feedback Generation
[![MIT License](https://github.com/user-attachments/assets/f0fc5011-9ca9-485f-bfb6-1661708a0be5)](https://raw.githubusercontent.com/Athira1979/Cross-Modal-Piano-Learning-Framework/refs/heads/main/LICENSE)
 
A state-of-the-art multimodal deep learning system for automatic piano skill assessment using audio, gesture, and posture data. Achieves robust classification of beginner/intermediate/advanced skill levels.
## ✨ Features
In this paper, the aim is to implement a method for enhancing piano skill development and participation. The existing methods have various drawbacks for feature extraction. This framework handles such problems by considering effective techniques. Its main contributions are summarized as follows,
+ Adaptive Wavelet-Enhanced Mel Frequency Cepstral Coefficients with Temporal Dynamics (AWavelet-MFCC-TD): This module uses integrated wavelet decomposition, MFCCs, and temporal dynamics to reduce noise.
+ Frequency-Domain Micro-Motion Analysis (FDMMA): This module uses an Adaptive Time-Frequency Temporal Convolutional Network (ATF-TCN) to accurately capture subtle finger and hand micro-motions.
+ Spatio-Temporal Attention-Driven Tracker (STAT): This module uses posture-aware temporal and spatial attention layers to track and evaluate body alignment during piano practice.
+ Cross-Modal Transformer with Modality-Specific Probing and Fusion (CMTPF): This module uses modality-specific probing to keep a balance between audio, gesture, and posture.
+ Real-Time Feedback Prioritization Network (RFP-Net): This module uses an encoder–decoder feedback system that delivers prioritized, actionable corrections on rhythm, fingering, and posture, providing adaptive guidance.
These contributions enhance piano skill development.
## 🚀 Quick Start
### 1. Clone & Install
```bash
https://github.com/Athira1979/Cross-Modal-Piano-Learning-Framework.git
cd Cross-Modal-Piano-Learning-Framework
pip install -r requirements.txt
```
### 2. Generate Dataset
```bash
python generate_dataset.py --num_sessions 1000 --split
```
### 3. Train Model
```
python train.py
```
### 4. Live Demo
```
# Webcam + Mic (default)
python demo.py live --width 1280 --height 720

# Web interface  
python demo.py web --share

# Simulation
python demo.py sim --duration 60
```
## 📊 Performance
```

```
## 🏗️ Architecture Overview
```
Audio (Wavelet+MFCC)  → AudioEncoder 
         │
Gesture   → FDMMA (TCN+ATFM) 
         │
Posture   → STABlock (PASA+PATA) 
         │
         ↓ CrossModal Fusion (CMTPF)
         │
          → Classifier  → [Beginner/Intermediate/Advanced]
```
## 🗂️ File Structure
```
dataset/
├── participant1/
│   └── session1/
│       ├── audio.npy      # Raw audio (16000Hz)
│       ├── gesture.npy    # (T, J, C) or (T, features)
│       ├── posture.npy    # (T, J, C) or (T, features)
│       └── meta.json      # {"skill_level": "beginner|intermediate|advanced"}
 
piano-ai/
├── 🎹 demo.py              # Live/Web demos
├── 📊 generate_dataset.py  # Synthetic data
├── 🏋️  train.py           # Training script
├── 📈 metrics.py          # Evaluation
├── data/dataset_loader.py # PyTorch Dataset
├── model.py              # PianoAI Model
├── config.py             # Hyperparams
├── dataset/              # Generated data
├── checkpoints/          # Trained models
└── docs/                 # Screenshots
 ```
## 🔧 Requirements
```
pip install torch torchvision torchaudio
pip install opencv-python mediapipe gradio pyaudio soundfile
pip install pandas numpy tqdm matplotlib seaborn scikit-learn
```
## 🎓 Training from Scratch
```
# 1. Generate data
python generate_dataset.py --num_sessions 5000 --realistic --split

# 2. Train  
python train.py

# 3. Evaluate
python metrics.py --predictions predictions.csv --plot
```
## 📄 License
<a href='https://raw.githubusercontent.com/Athira1979/Cross-Modal-Piano-Learning-Framework/refs/heads/main/LICENSE'>MIT License</a> - Free for commercial use!
