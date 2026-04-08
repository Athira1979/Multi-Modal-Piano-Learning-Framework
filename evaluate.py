# evaluate.py - FIXED ✅
import torch

def evaluate_model(model, loader, device):
    """
    Evaluate model and return true labels + predictions
    Fixed to handle 4-tuple data: (audio, hand, posture, label)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for audio, hand, posture, label in loader:  # ✅ Fixed: 4 items
            audio = audio.to(device)
            hand = hand.to(device)
            posture = posture.to(device)
            label = label.to(device)

            output = model(audio, hand, posture)  # ✅ Forward pass with all inputs
            _, preds = torch.max(output, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(label.cpu().tolist())

    return all_labels, all_preds
