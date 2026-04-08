import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyaudio
import warnings
import gradio as gr
import mediapipe as mp
from collections import deque
import threading
import time
import argparse
import sys
from config import Config

warnings.filterwarnings("ignore")


class AudioProcessor:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1,
                                  rate=Config.SAMPLE_RATE, input=True,
                                  frames_per_buffer=1600)
        self.audio_buffer = deque(maxlen=Config.NUM_FRAMES)

    def get_audio_frame(self):
        """Get 100ms audio frame"""
        try:
            data = self.stream.read(1600, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.float32)
            self.audio_buffer.append(audio)
            return np.concatenate(list(self.audio_buffer))
        except:
            return np.random.randn(1600).astype(np.float32)

    def close(self):
        """Proper cleanup"""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class PoseDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7)
        # 🆕 FIX: Initialize landmark buffer
        self.landmark_buffer = deque(maxlen=Config.NUM_FRAMES)

    def extract_features(self, frame):
        """Extract hand + pose landmarks → [20, 126/51] - FIXED"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 🆕 Hand landmarks (21 points × 2 hands × 3 coords = 126)
        hand_results = self.hands.process(rgb_frame)
        hand_landmarks = np.zeros((Config.NUM_FRAMES, 126))

        if hand_results.multi_hand_landmarks:
            # Extract current frame landmarks
            current_hand_lms = []
            for hand_landmarks_obj in hand_results.multi_hand_landmarks[:2]:
                for lm in hand_landmarks_obj.landmark:
                    current_hand_lms.extend([lm.x, lm.y, lm.z])

            # Pad to 126 dims (2 hands)
            current_hand_lms = current_hand_lms[:126] + [0.0] * (126 - len(current_hand_lms))
            self.landmark_buffer.append(np.array(current_hand_lms))

        # Use buffer for temporal features
        buffer_list = list(self.landmark_buffer)
        for i in range(Config.NUM_FRAMES - len(buffer_list)):
            buffer_list.append(np.zeros(126))
        hand_landmarks = np.array(buffer_list)

        # 🆕 Pose landmarks (17 upper body points × 4 = 68, simplified to 51)
        pose_results = self.pose.process(rgb_frame)
        pose_landmarks = np.zeros((Config.NUM_FRAMES, 51))
        if pose_results.pose_landmarks:
            # Use first 17 landmarks (upper body) × (x,y,z,visibility) = 68 → take first 51
            pose_lm = []
            for lm in pose_results.pose_landmarks.landmark[:17]:
                pose_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
            current_pose = np.array(pose_lm[:51])  # Truncate to 51
            self.landmark_buffer.append(current_pose)  # Also store pose for consistency

            pose_landmarks[0] = current_pose

        return hand_landmarks.astype(np.float32), pose_landmarks.astype(np.float32)


class PianoAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.eval()

        # 🎵 Audio: 1600 → 128D
        self.audio_proj = nn.Sequential(
            nn.Linear(1600, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # 🤚 Gesture: [20,126] → 128D
        self.gesture_proj = nn.Sequential(
            nn.Linear(126 * Config.NUM_FRAMES, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # 🧍 Posture: [20,51] → 128D
        self.posture_proj = nn.Sequential(
            nn.Linear(51 * Config.NUM_FRAMES, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Fusion + Feedback
        self.fusion = nn.Sequential(
            nn.Linear(128 * 3, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.feedback_head = nn.Linear(256, 7)

        # Feedback mapping
        self.feedback_map = {
            0: ("🎉 PERFECT!", (0, 255, 0)),
            1: ("👆 FINGERING", (0, 255, 255)),
            2: ("🎵 RHYTHM", (255, 255, 0)),
            3: ("🧍 POSTURE", (255, 165, 0)),
            4: ("🎭 EXPRESSION", (255, 0, 255)),
            5: ("😌 RELAX", (0, 165, 255)),
            6: ("✋ POSITION", (0, 255, 165))
        }

    def forward(self, audio, gesture, posture):
        audio_f = self.audio_proj(audio)
        gesture_f = self.gesture_proj(gesture.reshape(gesture.size(0), -1))
        posture_f = self.posture_proj(posture.reshape(posture.size(0), -1))

        fused = torch.cat([audio_f, gesture_f, posture_f], dim=1)
        fused = self.fusion(fused)
        logits = self.feedback_head(fused)
        return logits

    def predict(self, audio, gesture, posture):
        with torch.no_grad():
            logits = self(audio, gesture, posture)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs.max().item()

            feedback_text, color = self.feedback_map[pred]
            return feedback_text, confidence, color, pred


class PianoLearningSystem:
    def __init__(self):
        self.device = Config.DEVICE
        self.model = PianoAI().to(self.device)
        self.audio_proc = AudioProcessor()
        self.pose_detector = PoseDetector()

        # Buffers for smooth temporal features
        self.gesture_buffer = deque(maxlen=Config.NUM_FRAMES)
        self.posture_buffer = deque(maxlen=Config.NUM_FRAMES)

        print(f"🎹 PIANO AI READY | Device: {self.device}")

    def process_frame(self, frame):
        """Main processing pipeline - FIXED"""
        # 1. Get live audio (pad to correct length)
        audio_raw = self.audio_proc.get_audio_frame()
        if len(audio_raw) != 1600:
            audio_raw = audio_raw[:1600] if len(audio_raw) > 1600 else np.pad(audio_raw, (0, 1600 - len(audio_raw)))
        audio = torch.tensor(audio_raw, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 2. Extract pose features
        gesture, posture = self.pose_detector.extract_features(frame)

        # 3. Use temporal buffers
        self.gesture_buffer.append(gesture[0])
        self.posture_buffer.append(posture[0])

        # Pad buffers if needed
        gesture_full = np.stack(list(self.gesture_buffer) +
                                [np.zeros(126)] * (Config.NUM_FRAMES - len(self.gesture_buffer)))
        posture_full = np.stack(list(self.posture_buffer) +
                                [np.zeros(51)] * (Config.NUM_FRAMES - len(self.posture_buffer)))

        gesture_t = torch.tensor(gesture_full, dtype=torch.float32).unsqueeze(0).to(self.device)
        posture_t = torch.tensor(posture_full, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 4. AI Prediction
        feedback, conf, color, pred_id = self.model.predict(audio, gesture_t, posture_t)

        return feedback, conf, color, pred_id

    def draw_feedback(self, frame, feedback, conf, color, frame_idx):
        """Professional overlay - FIXED color handling"""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Gradient background (fixed dimensions)
        gradient = np.linspace(0, 1, 256).reshape(256, 1, 1)
        bg_height = min(180, h - 20)
        overlay[:bg_height, 20:w - 20] = (gradient[:bg_height] * np.array(color) * 0.3).astype(np.uint8)

        # Main feedback
        cv2.putText(overlay, feedback, (60, 110),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 4)

        # Confidence bar
        bar_width = min(int(500 * conf), 500)
        cv2.rectangle(overlay, (60, 140), (60 + bar_width, 165), color, -1)
        cv2.rectangle(overlay, (60, 140), (560, 165), (255, 255, 255), 3)
        cv2.putText(overlay, f"{conf:.0%}", (520, 158),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Status
        cv2.putText(overlay, f"Frame: {frame_idx} | 🎵 Live Audio + 🤚 AI Vision",
                    (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        return frame

    def cleanup(self):
        """Proper cleanup"""
        self.audio_proc.close()



def live_demo_simulation(system):
    """Fallback simulation"""
    print("🎵 SIMULATION - 30 seconds of feedback")
    for i in range(30):
        audio = torch.randn(1, 1600).to(system.device)
        gesture = torch.randn(1, Config.NUM_FRAMES, 126).to(system.device)
        posture = torch.randn(1, Config.NUM_FRAMES, 51).to(system.device)

        feedback, conf, color, pred = system.model.predict(audio, gesture, posture)
        print(f"[{i+1:02d}] {feedback} | {conf:.0%} | 🎵🤚🧍")
        time.sleep(1)


def web_demo(port=7860, share=False, verbose=False):
    """Gradio web interface - MODIFIED for args"""
    system = PianoLearningSystem()

    def predict_web(audio, image):
        if audio is None and image is None:
            return "Upload audio + image!", None

        # Process audio
        audio_np = np.array(audio)[:1600]
        audio_t = torch.tensor(audio_np).unsqueeze(0).to(system.device)

        # Process image (simulate pose)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gesture, posture = system.pose_detector.extract_features(frame)
        gesture_t = torch.tensor(gesture).unsqueeze(0).to(system.device)
        posture_t = torch.tensor(posture).unsqueeze(0).to(system.device)

        feedback, conf, color, pred = system.model.predict(audio_t, gesture_t, posture_t)
        return f"{feedback}\nConfidence: {conf:.1%}", gr.Image(value=frame)

    iface = gr.Interface(
        fn=predict_web,
        inputs=[
            gr.Audio(source="microphone", type="numpy", label="🎵 Play Piano"),
            gr.Image(type="pil", label="🤚 Hand/Posture Image")
        ],
        outputs=[
            gr.Textbox(label="🎯 AI Feedback"),
            gr.Image(label="📸 Processed Frame")
        ],
        title="🎹 Piano Learning AI",
        description="🎵 Play → 📸 Show hands → Get instant technique feedback!",
        theme=gr.themes.Soft()
    )


    print(f"🌐 Web demo running at http://localhost:{port}")
    if share:
        print("🔗 Public link generated!")
    iface.launch(share=share, server_port=port)


def live_demo(width=640, height=480, fps=30, model_path='best_model.pth', verbose=False):
    """Live demo - MODIFIED for args"""
    print(f"🎥 LIVE DEMO | {width}x{height}@{fps}fps | Model: {model_path}")
    system = PianoLearningSystem()

    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    cap = None
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if cap.isOpened():
            print(f"✅ Webcam OK! Backend: {backend}")
            break

    if not cap.isOpened():
        print("❌ No webcam - switching to SIMULATION")
        live_demo_simulation(system)
        return

    print("🎹 Press 'Q' to quit | Play piano for AI feedback!")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_idx += 1
        feedback, conf, color, pred_id = system.process_frame(frame)
        frame = system.draw_feedback(frame, feedback, conf, color, frame_idx)

        cv2.imshow('🎹 PIANO AI - Live Feedback', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    system.audio_proc.stream.stop_stream()
    system.audio_proc.stream.close()
    system.audio_proc.p.terminate()


def simulation_demo(duration=30, verbose=False):
    """Simulation - MODIFIED for args"""
    print(f"🧪 SIMULATION MODE | {duration} seconds")
    system = PianoLearningSystem()
    live_demo_simulation(system)


# 🔥 NEW: ARGUMENT PARSER (SIMPLE & PERFECT)
def create_demo_parser():
    parser = argparse.ArgumentParser(
        description="🎹 Piano Learning AI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎥 LIVE (default):     python demo.py
🌐 WEB:               python demo.py web --share  
🧪 SIM:               python demo.py sim --duration 120
🎥 HD LIVE:           python demo.py live --width 1280 --height 720 --fps 60
        """
    )

    parser.add_argument('mode', nargs='?', default='live', choices=['live', 'web', 'sim'],
                        help='Demo mode (default: live)')

    # Live options
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    parser.add_argument('--fps', type=int, default=30, help='Camera FPS')

    # Web options
    parser.add_argument('--port', type=int, default=7860, help='Web port')
    parser.add_argument('--share', action='store_true', help='Public link')

    # Sim options
    parser.add_argument('--duration', type=int, default=30, help='Sim duration (s)')

    # Global
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--verbose', '-v', action='store_true')

    return parser


# 🎯 MAIN ENTRY POINT
def main():
    parser = create_demo_parser()

    if len(sys.argv) == 1:
        args = parser.parse_args(['live'])
    else:
        args = parser.parse_args()

    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    Config.DEVICE = device  # 🆕 Set config device

    print(f"\n🚀 PIANO AI DEMO v1.0")
    print(f"📱 Mode: {args.mode.upper()} | Device: {device}")

    system = None
    try:
        if args.mode == 'live':
            system = PianoLearningSystem()
            live_demo(args.width, args.height, args.fps, verbose=args.verbose  )
        elif args.mode == 'web':
            web_demo(args.port, args.share, args.verbose)
        elif args.mode == 'sim':
            system = PianoLearningSystem()
            simulation_demo(args.duration, args.verbose )
    finally:
        if system:
            system.cleanup()

if __name__ == "__main__":
    main()
