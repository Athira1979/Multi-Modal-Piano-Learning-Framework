 
import numpy as np
import torch
import torch.nn as nn
from models.awavelet_mfcc_td import AWaveletMFCC_TD
from models.fdmma import FDMMA, AMR_STFT
from models.stat import STATBlock
from models.cmtpf import CMTPF
from models.rfp_net import RFPNet


class PianoAIModel(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        # Feature extractors (non-trainable)
        self.audio_extractor = AWaveletMFCC_TD()
        self.motion_extractor = AMR_STFT(sr=120)  # ✅ FIXED: match hand FPS

        # Learned modules
        self.fdmma = FDMMA(input_dim=33)
        self.stat = STATBlock(dim=128)
        self.cmtpf = CMTPF(da=128, dg=128, dp=128, d_model=128)
        self.rfp = RFPNet(dim=128)

        # Projections - ✅ FIXED DIMENSIONS
        self.audio_proj = nn.Sequential(
            nn.Linear(75, 128),
            nn.ReLU(),
            nn.Dropout(0.3)) # AWaveletMFCC_TD output
        self.motion_proj = nn.Sequential(
            nn.Linear(33, 128),
            nn.ReLU(),
            nn.Dropout(0.3))  # STFT features
        self.posture_proj = nn.Sequential(
            nn.Linear(30, 128),
            nn.ReLU(),
            nn.Dropout(0.3)) # 10 joints * 3 coords

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128)
        )
    def forward(self, audio_signal, motion_signal, posture_signal):
        B = audio_signal.shape[0]
        # -------------------------------
        # 1. AUDIO FEATURES (BATCHED)
        # -------------------------------
        audio_feats = []
        specs = []
        for b in range(B):
            # Original audio features
            audio_feat = self.audio_extractor.extract_features(
                audio_signal[b].cpu().numpy()
            )
            audio_feats.append(audio_feat)
            # Spectrogram for FDMMA
            spec = self.motion_extractor.transform(audio_signal[b].cpu().numpy())
            specs.append(spec.T)
        max_len = max(f.shape[0] for f in audio_feats)
        padded = []
        for f in audio_feats:
            pad_len = max_len - f.shape[0]
            if pad_len > 0:
                f = np.pad(f, ((0, pad_len), (0, 0)))
            padded.append(f)
        audio_feat = torch.tensor(np.array(padded), dtype=torch.float32).to(self.device)
        # -------------------------------
        # 2. MOTION FEATURES (BATCHED)
        # -------------------------------
        motion_feats = []
        for b in range(B):
            hand_np = motion_signal[b].cpu().numpy()  # (T, 42, 3)
            # Flatten spatial → 1D temporal signal
            hand_1d = hand_np.reshape(hand_np.shape[0], -1).mean(axis=1)  # (T,)
            motion_spec = self.motion_extractor.transform(hand_1d)
            motion_feats.append(motion_spec.T)
        motion_feat = torch.tensor(np.array(motion_feats), dtype=torch.float32).to(self.device)
        # -------------------------------
        # 3. PROJECTIONS
        # -------------------------------
        audio_seq = self.audio_proj(audio_feat)  # (B, 128)
        motion_seq = self.motion_proj(motion_feat)  # (B, 128)
        posture_feat = posture_signal.unsqueeze(1)  # (B, 1, 30)
        posture_seq = self.posture_proj(posture_feat)  # (B, 1, 128)
        # -------------------------------
        # 4. CMTPF FUSION (CORE)
        # -------------------------------
        # Add temporal dimension for CMTPF
        T = audio_seq.shape[1]
        motion_seq = motion_seq[:, :T]
        posture_seq = posture_seq.expand(-1, T, -1)
        fused = self.cmtpf(audio_seq, motion_seq, posture_seq)

        fused = self.dropout(fused)  # 👈 ADD THIS

        output = self.classifier(fused)
        # -------------------------------
        # 5. RFP FINAL PREDICTION
        # -------------------------------
        output = self.rfp(output)  # (B, T, 3)
        return output.mean(dim=1)  # (B, 3) final logits
