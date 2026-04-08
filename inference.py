 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data.dataset_loader import PianoDataset   
from model import PianoAIModel
import numpy as np
import argparse

class PianoInference:
    def __init__(self, model_path="checkpoints/best_model.pth", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Inference on: {self.device}")
        
        # Load model
        self.model = PianoAIModel(self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ Model loaded from {model_path}")
    
    def predict_single(self, audio, hand, posture):
        """
        Predict on single sample
        audio, hand, posture: tensors of shape [seq_len, features]
        Returns: predicted class (int), probabilities (tensor)
        """
        with torch.no_grad():
            audio = audio.unsqueeze(0).to(self.device)  # [1, seq, feat]
            hand = hand.unsqueeze(0).to(self.device)
            posture = posture.unsqueeze(0).to(self.device)
            
            output = self.model(audio, hand, posture)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            
        return pred, probs.cpu().numpy()[0]
    
    def predict_batch(self, audio_batch, hand_batch, posture_batch):
        """Predict on batch"""
        dataset = TensorDataset(audio_batch, hand_batch, posture_batch)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_preds, all_probs = [], []
        with torch.no_grad():
            for audio, hand, posture in loader:
                output = self.model(audio.to(self.device), 
                                  hand.to(self.device), 
                                  posture.to(self.device))
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)
    
    def predict_from_files(self, audio_file, hand_file, posture_file):
        """Predict from raw data files (customize paths)"""
        # Load your raw data here
        audio = torch.tensor(np.load(audio_file)).float()
        hand = torch.tensor(np.load(hand_file)).float() 
        posture = torch.tensor(np.load(posture_file)).float()
        
        pred, probs = self.predict_single(audio, hand, posture)
        return pred, probs

# CLI Usage
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/best_model.pth")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--hand", required=True)
    parser.add_argument("--posture", required=True)
    args = parser.parse_args()
    
    infer = PianoInference(args.model)
    pred, probs = infer.predict_from_files(args.audio, args.hand, args.posture)
    
    print(f"🎹 Prediction: {pred}")
    print(f"Probabilities: {probs}")
    print(f"Confidence: {max(probs):.3f}")

if __name__ == "__main__":
    main()
