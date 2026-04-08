# generate_dataset.py
import os
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import random
import argparse
import json
from pathlib import Path


# ==============================
# CONFIGURATION
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="🎹 Generate Piano Learning Dataset")
    parser.add_argument('--num_sessions', type=int, default=1500, help='Number of sessions')
    parser.add_argument('--dataset_path', type=str, default='dataset', help='Output directory')
    parser.add_argument('--min_duration', type=int, default=60, help='Min session duration (s)')
    parser.add_argument('--max_duration', type=int, default=180, help='Max session duration (s)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--realistic', action='store_true', help='More realistic variations')
    parser.add_argument('--split', action='store_true', help='Create train/val/test splits')

    return parser.parse_args()


# ==============================
# SETUP
# ==============================
def setup_config(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = {
        'AUDIO_SR': 16000,
        'HAND_FPS': 120,
        'POSTURE_FPS': 30,
        'NUM_JOINTS_HAND': 42,  # 21 joints × 2 hands
        'NUM_JOINTS_POSTURE': 17,  # Upper body
    }

    return config


# ==============================
# DIRECTORIES
# ==============================
def create_directories(dataset_path):
    dirs = ['audio', 'hand', 'posture', 'labels']
    for dir_name in dirs:
        os.makedirs(f"{dataset_path}/{dir_name}", exist_ok=True)
    print(f"📁 Created directories in {dataset_path}/")


# ==============================
# PARTICIPANTS & LABELS
# ==============================
def create_participants(num_participants=30):
    participants = []
    for i in range(num_participants):
        if i < 12:
            skill = "beginner"
            error_rate = random.uniform(0.25, 0.40)
        elif i < 22:
            skill = "intermediate"
            error_rate = random.uniform(0.10, 0.25)
        else:
            skill = "advanced"
            error_rate = random.uniform(0.00, 0.10)

        participants.append({
            "id": i,
            "age": random.randint(18, 35),
            "skill": skill,
            "gender": random.choice(["male", "female"]),
            "error_rate": error_rate
        })
    return participants


# 7 feedback categories matching the model
FEEDBACK_CATEGORIES = [
    "PERFECT", "FINGERING", "RHYTHM", "POSTURE",
    "EXPRESSION", "RELAX", "POSITION"
]


# ==============================
# REALISTIC GENERATION
# ==============================
def generate_audio(duration, skill, activity, error_rate=0.2):
    """Generate realistic piano audio with skill-based variations"""
    t = np.linspace(0, duration, int(config['AUDIO_SR'] * duration))

    # Base piano frequencies (C4, D4, E4, F4, G4, A4)
    piano_notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00]

    # Skill-based note patterns
    if skill == "beginner":
        tempo_variation = random.uniform(0.7, 1.3)
        note_errors = int(duration * error_rate * 2)
    elif skill == "intermediate":
        tempo_variation = random.uniform(0.85, 1.15)
        note_errors = int(duration * error_rate * 1)
    else:
        tempo_variation = random.uniform(0.95, 1.05)
        note_errors = int(duration * error_rate * 0.5)

    # Generate note sequence
    signal = np.zeros_like(t)
    note_duration = 0.3 * tempo_variation
    notes_per_second = 1 / note_duration

    for i in range(int(duration * notes_per_second)):
        start_time = i * note_duration
        end_time = min((i + 1) * note_duration, duration)

        # Add wrong notes based on error_rate
        if random.random() < error_rate and i < note_errors:
            freq = random.choice(piano_notes)
        else:
            freq = random.choice(piano_notes[:4])  # Prefer simpler notes

        note_t = t[(t >= start_time) & (t < end_time)]
        if len(note_t) > 0:
            signal[(t >= start_time) & (t < end_time)] = 0.4 * np.sin(2 * np.pi * freq * note_t)

    # Add realistic noise/reverb
    noise = np.random.normal(0, 0.02 * (0.3 if skill == "beginner" else 0.01), t.shape)
    signal += noise

    return np.clip(signal, -1.0, 1.0).astype(np.float32)


def generate_hand_data(duration, skill, primary_error):
    """Generate hand landmarks with skill-specific errors"""
    frames = int(duration * config['HAND_FPS'])
    joints = config['NUM_JOINTS_HAND']  # 42

    # Base hand position (piano keyboard area)
    base_pos = np.array([0.4, 0.6, 0.0])  # x,y,z normalized

    data = np.tile(base_pos, (frames, joints // 3, 1)) + np.random.randn(frames, joints // 3, 3) * 0.02

    # Add finger motion
    finger_joints = slice(12, 30)  # Thumb to pinky
    for f in range(frames):
        if random.random() < 0.3:  # Occasional finger movement
            data[f, finger_joints // 3, 1] += np.sin(f * 0.1) * 0.05

    # Skill-based errors
    if skill != "advanced":
        error_frames = int(frames * 0.1)
        for _ in range(error_frames):
            frame = random.randint(0, frames - 1)
            if primary_error == "FINGERING":
                data[frame, finger_joints // 3, :] += np.random.randn(*data[frame, finger_joints // 3, :].shape) * 0.1
            elif primary_error == "POSITION":
                data[frame, :, 0] += np.random.normal(0, 0.08, joints // 3)

    return data.reshape(frames, -1)  # (T, 126)


def generate_posture_data(duration, skill, primary_error):
    """Generate posture data with realistic variations"""
    frames = int(duration * config['POSTURE_FPS'])
    joints = config['NUM_JOINTS_POSTURE']

    # Base upright posture
    base_posture = np.zeros((joints, 3))
    base_posture[:, 1] = np.linspace(0.2, 0.8, joints)  # Y positions

    data = np.tile(base_posture, (frames, 1, 1)) + np.random.randn(frames, joints, 3) * 0.01

    # Add breathing motion
    for f in range(frames):
        data[f, 5:10, 2] += 0.02 * np.sin(f * 0.2)  # Chest breathing

    # Posture errors
    if skill != "advanced" and primary_error == "POSTURE":
        error_frames = int(frames * 0.15)
        for _ in range(error_frames):
            frame = random.randint(0, frames - 1)
            data[frame, 2:8, 1] -= 0.1  # Slouch shoulders

    return data.reshape(frames, -1)[:frames, :51]  # (T, 51) matching model


# ==============================
# MAIN GENERATION
# ==============================
def main():
    args = parse_args()
    global config
    config = setup_config(args)

    print(f"🚀 Generating {args.num_sessions} sessions...")
    print(f"📁 Dataset: {args.dataset_path}")

    create_directories(args.dataset_path)
    participants = create_participants()

    metadata = []

    for i in tqdm(range(args.num_sessions), desc="Sessions"):
        session_id = f"{i:04d}"

        participant = random.choice(participants)
        duration = random.randint(args.min_duration, args.max_duration)
        activity = random.choice(["scales", "chords", "pieces"])

        # 🆕 Primary feedback label (most common error)
        if participant["skill"] == "advanced":
            primary_label = random.choice(["PERFECT"] + FEEDBACK_CATEGORIES[1:])
            label_dist = [0.7 if x == primary_label else 0.04 for x in FEEDBACK_CATEGORIES]
        else:
            primary_label = random.choices(FEEDBACK_CATEGORIES[1:], weights=[0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.15])[0]
            label_dist = [0.05 if x == "PERFECT" else 0.12 if x == primary_label else 0.11 for x in FEEDBACK_CATEGORIES]

        primary_error = primary_label

        # Generate data
        audio = generate_audio(duration, participant["skill"], activity, participant["error_rate"])
        hand_data = generate_hand_data(duration, participant["skill"], primary_error)
        posture_data = generate_posture_data(duration, participant["skill"], primary_error)

        # Save files
        audio_path = f"{args.dataset_path}/audio/session_{session_id}.wav"
        hand_path = f"{args.dataset_path}/hand/session_{session_id}.npy"
        posture_path = f"{args.dataset_path}/posture/session_{session_id}.npy"
        label_path = f"{args.dataset_path}/labels/session_{session_id}.json"

        sf.write(audio_path, audio, config['AUDIO_SR'])
        np.save(hand_path, hand_data)
        np.save(posture_path, posture_data)

        # Save per-session labels
        labels = {
            "primary_feedback": primary_label,
            "label_distribution": label_dist,
            "session_info": {
                "participant_id": participant["id"],
                "skill": participant["skill"],
                "activity": activity,
                "duration": duration
            }
        }
        with open(label_path, 'w') as f:
            json.dump(labels, f)

        metadata.append({
            "session": session_id,
            "participant_id": participant["id"],
            "age": participant["age"],
            "gender": participant["gender"],
            "skill": participant["skill"],
            "activity": activity,
            "duration_sec": duration,
            "primary_feedback": primary_label,
            "audio_path": audio_path,
            "hand_path": hand_path,
            "posture_path": posture_path,
            "label_path": label_path
        })

    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv(f"{args.dataset_path}/metadata.csv", index=False)

    # 🆕 Dataset splits
    if args.split:
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=df['primary_feedback'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed,
                                           stratify=temp_df['primary_feedback'])

        train_df.to_csv(f"{args.dataset_path}/train.csv", index=False)
        val_df.to_csv(f"{args.dataset_path}/val.csv", index=False)
        test_df.to_csv(f"{args.dataset_path}/test.csv", index=False)
        print("✅ Train/Val/Test splits created!")

    # Stats
    print(f"\n📊 Dataset Stats:")
    print(df['skill'].value_counts())
    print(df['primary_feedback'].value_counts())
    print(f"💾 Total size: {len(df)} sessions")
    print("✅ Dataset generation complete!")


if __name__ == "__main__":
    main()
