#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load LLaVA-NeXT model utilities
from llava.model.builder import load_pretrained_model

# ===== Dataset Class for Training/Validation =====
class DisasterImageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.event_categories = {}
        # Map event name to numeric label
        for idx, event in enumerate(sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])):
            self.event_categories[event] = idx
        for event, label in self.event_categories.items():
            image_dir = os.path.join(data_path, event, "images")
            if os.path.exists(image_dir):
                for file in os.listdir(image_dir):
                    if file.startswith("._"):
                        continue
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(image_dir, file))
                        self.labels.append(label)
        print(f"Loaded {len(self.image_paths)} images from {data_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ===== Define Image Transformations =====
transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
])

# ===== Device Selection =====
def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load LLaVA Vision Encoder =====
def load_llava_vision_encoder():
    pretrained = "lmms-lab/llama3-llava-next-8b"  # adjust if needed
    model_name = "llava_llama3"
    device_map = "auto"
    # Load model utilities: returns (tokenizer, model, image_processor, max_length)
    _, full_model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map=device_map)
    full_model.to("cpu")
    full_model.eval()
    try:
        vision_encoder = full_model.get_vision_tower()
    except AttributeError:
        vision_encoder = full_model.vision_model
    vision_encoder.to("cpu")
    vision_encoder.eval()
    for param in vision_encoder.parameters():
        param.requires_grad = False
    return vision_encoder, image_processor

# ===== Define the Classifier Head =====
class LlavaNextClassifier(nn.Module):
    def __init__(self, vision_encoder, num_classes, device):
        super().__init__()
        self.vision_encoder = vision_encoder
        # Run dummy input to determine feature dimension
        dummy = torch.zeros((1, 3, 336, 336)).to(device)
        with torch.no_grad():
            out = self.vision_encoder(dummy)
        if hasattr(out, "pooler_output"):
            feat_dim = out.pooler_output.shape[-1]
        else:
            feat_dim = out[:, 0, :].shape[-1]
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
    def forward(self, images):
        feats = self.vision_encoder(images)
        if hasattr(feats, "pooler_output"):
            pooled = feats.pooler_output
        else:
            pooled = feats[:, 0, :]
        logits = self.classifier(pooled)
        return logits

# ===== Compute Accuracy Helper =====
def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

# ===== Training Loop with Checkpointing and Early Stopping =====
def train_model(model, train_loader, val_loader, device, epochs=10, lr=2e-5, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_acc = 0.0
    no_improve_count = 0
    best_model_path = "llava_next_classifier_best.pth"
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_loss, train_acc = compute_accuracy(model, train_loader, device)
        val_loss, val_acc = compute_accuracy(model, val_loader, device)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved new best model at epoch {epoch+1} with Val Acc {best_acc:.4f}")
        else:
            no_improve_count += 1
            print(f"  No improvement for {no_improve_count} epochs.")
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement for {patience} consecutive epochs.")
                break
    model.load_state_dict(torch.load(best_model_path))
    return model

def main():
    parser = argparse.ArgumentParser(description="Train LLaVA-NeXT Classifier for Disaster Events")
    parser.add_argument("--train_dir", type=str, default="split_dataset/train", help="Path to training data")
    parser.add_argument("--val_dir", type=str, default="split_dataset/test", help="Path to validation data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    args = parser.parse_args()
    
    device = select_device()
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = DisasterImageDataset(args.train_dir, transform=transform)
    val_dataset = DisasterImageDataset(args.val_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    num_classes = len(train_dataset.event_categories)
    print(f"Detected {num_classes} classes: {train_dataset.event_categories}")
    
    vision_encoder, image_processor = load_llava_vision_encoder()
    classifier_model = LlavaNextClassifier(vision_encoder, num_classes, device)
    classifier_model.to(device)
    
    # Train model with checkpointing and early stopping
    classifier_model = train_model(classifier_model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, patience=3)
    
    torch.save(classifier_model.state_dict(), "llava_next_classifier.pth")
    print("Saved fine-tuned model as llava_next_classifier.pth")
    
    train_loss, train_acc = compute_accuracy(classifier_model, train_loader, device)
    val_loss, val_acc = compute_accuracy(classifier_model, val_loader, device)
    print(f"\nFinal Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
    print(f"Final Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
