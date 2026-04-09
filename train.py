
# train.py - IMPROVED VERSION
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data.dataset_loader import PianoDataset
from tqdm import tqdm
from evaluate import evaluate_model
from model import PianoAIModel
from utils.metrics import compute_metrics
from utils.plot_metrics import plot_training_curves
import matplotlib.pyplot as plt
import os

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for audio, hand, posture, label in val_loader:
            audio, hand, posture, label = (
                audio.to(device), hand.to(device),
                posture.to(device), label.to(device)
            )

            output = model(audio, hand, posture)
            loss = criterion(output, label)

            val_loss += loss.item()
            _, preds = torch.max(output, 1)
            val_correct += (preds == label).sum().item()
            val_total += label.size(0)

    return val_loss / len(val_loader), val_correct / val_total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Create output directory
    os.makedirs("checkpoints", exist_ok=True)

    # -------------------
    # DATA
    # -------------------
    dataset = PianoDataset("dataset/metadata.csv", max_len=5)
    print(f"📊 Dataset size: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=False)

    # -------------------
    # MODEL & OPTIMIZER
    # -------------------
    model = PianoAIModel(device).to(device)

    # Weight initialization
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # -------------------
    # TRAINING LOOP WITH EARLY STOPPING
    # -------------------
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(50):  # Increased epochs
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/50")
        for audio, hand, posture, label in progress:
            audio, hand, posture, label = (
                audio.to(device), hand.to(device),
                posture.to(device), label.to(device)
            )

            optimizer.zero_grad()
            output = model(audio, hand, posture)
            loss = criterion(output, label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(output, 1)
            correct += (preds == label).sum().item()
            total += label.size(0)

            progress.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{correct / total:.4f}"
            })

        scheduler.step()

        # Validation
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        val_metrics = validate(model, val_loader, criterion, device)
        val_loss, val_acc = val_metrics

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, "checkpoints/best_model.pth")
            patience_counter = 0
            print(f"💾 New best model saved! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        print(f"\n📈 Epoch {epoch + 1}/50")
        print(f"   Train: Loss={train_loss:.4f} | Acc={train_acc:.4f}")
        print(f"   Val:   Loss={val_loss:.4f} | Acc={val_acc:.4f}")
        print(f"   LR: {scheduler.get_last_lr()[0]:.2e}")

        # Early stopping
        if patience_counter >= patience:
            print(f"⏹️  Early stopping at epoch {epoch + 1}")
            break

    # Load best model for evaluation
    checkpoint = torch.load("checkpoints/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    print("\n🔍 Final Evaluation...")
    num_classes = 8
    y_true, y_pred = evaluate_model(model, val_loader, device)
    metrics, cm = compute_metrics(y_true, y_pred, num_classes)

    # Save metrics
    df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    df.to_csv("evaluation_metrics.csv", index=False)
    print("\n📊 Final Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")
        
    plot_training_curves(train_losses, val_losses, train_accs, val_accs) 
 
 

if __name__ == '__main__':
    main()
