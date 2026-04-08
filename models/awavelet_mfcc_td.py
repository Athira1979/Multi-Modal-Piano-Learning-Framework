import torch
import torch.nn as nn
import numpy as np
import librosa
import pywt
from scipy.fftpack import dct

class AWaveletMFCC_TD:
    def __init__(self, sr=16000, frame_size=0.025, frame_stride=0.01, n_mfcc=13):
        self.sr = sr
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.n_mfcc = n_mfcc

    # -------------------------------
    # 1. PRE-EMPHASIS
    # -------------------------------
    def pre_emphasis(self, signal, alpha=0.97):
        return np.append(signal[0], signal[1:] - alpha * signal[:-1])

    # -------------------------------
    # 2. FRAMING
    # -------------------------------
    def framing(self, signal):
        frame_length = int(self.frame_size * self.sr)
        frame_step = int(self.frame_stride * self.sr)

        signal_length = len(signal)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(signal, z)

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

        return pad_signal[indices.astype(np.int32, copy=False)]

    # -------------------------------
    # 3. HAMMING WINDOW
    # -------------------------------
    def windowing(self, frames):
        return frames * np.hamming(frames.shape[1])

    # -------------------------------
    # 4. ADAPTIVE WAVELET TRANSFORM
    # -------------------------------
    def wavelet_transform(self, frames):
        wavelet_features = []

        for frame in frames:
            # Adaptive choice (simple version)
            wavelet = 'db4' if np.var(frame) > 0.01 else 'haar'

            coeffs = pywt.wavedec(frame, wavelet, level=3)

            # Extract statistical features per subband
            features = []
            for c in coeffs:
                features.extend([
                    np.mean(c),
                    np.std(c),
                    np.max(c)
                ])

            wavelet_features.append(features)

        return np.array(wavelet_features)

    # -------------------------------
    # 5. MFCC + AODCT
    # -------------------------------
    def mfcc_extraction(self, signal):
        if hasattr(signal, "detach"):
            signal = signal.detach().cpu().numpy()

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            hop_length=int(self.frame_stride * self.sr),
            n_fft=int(self.frame_size * self.sr)
        )

        return mfcc.T  # ✅ (T, 13)


    # -------------------------------
    # 6. DELTA FEATURES
    # -------------------------------
    def temporal_dynamics(self, features):
        delta = librosa.feature.delta(features.T)
        delta2 = librosa.feature.delta(features.T, order=2)

        return np.hstack([
            features,
            delta.T,
            delta2.T
        ])

    # -------------------------------
    # 7. FULL PIPELINE
    # -------------------------------
    def extract_features(self, signal):
        # Step 1: Pre-emphasis
        emphasized = self.pre_emphasis(signal)

        # Step 2: Framing
        frames = self.framing(emphasized)

        # Step 3: Windowing
        windowed = self.windowing(frames)

        # Step 4: Wavelet features
        wavelet_feat = self.wavelet_transform(windowed)

        # Step 5: MFCC + AODCT
        mfcc_feat = self.mfcc_extraction(signal)

        # Align lengths
        min_len = min(len(wavelet_feat), len(mfcc_feat))
        wavelet_feat = wavelet_feat[:min_len]
        mfcc_feat = mfcc_feat[:min_len]

        # Step 6: Fusion
        fused = np.hstack([mfcc_feat, wavelet_feat])

        # Step 7: Temporal dynamics
        final_features = self.temporal_dynamics(fused)

        return final_features
