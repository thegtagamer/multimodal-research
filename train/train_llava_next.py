import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load LLaVA-NeXT model utilities
from llava.model.builder import load_pretrained_model

# ===== Dataset Class =====
class DisasterImageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.event_categories = {}
        for idx, event in enumerate(sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])):
            self.event_categories[event] = idx
        for event, label in self.event_categories.items():
            image_dir = os.path.join(data_path, event, "images")
            if os.path.exists(image_dir):
                for file in os.listdir(image_dir):
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

# ===== Load Pretrained LLaVA-NeXT Model =====
pretrained = "lmms-lab/llama3-llava-next-8b"  # adjust if needed
model_name = "llava_llama3"
device = "cuda" if torch.cuda.is_available() else "cpu"
device_map = "auto"
# This returns (tokenizer, model, image_processor, max_length)
_, full_model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map=device_map)
full_model.to(device)
full_model.eval()  # Initialize all modules

# ===== Extract the Vision Encoder =====
try:
    vision_encoder = full_model.get_vision_tower()
except AttributeError:
    vision_encoder = full_model.vision_model
vision_encoder.to(device)
vision_encoder.eval()
for param in vision_encoder.parameters():
    param.requires_grad = False

# ===== Define the Classifier Head =====
class LlavaNextClassifier(nn.Module):
    def __init__(self, vision_encoder, num_classes, device):
        super().__init__()
        self.vision_encoder = vision_encoder
        # Run a dummy input to determine feature dimension
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

# ===== Image Transformations =====
transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
])

# ===== Device Selection (optional interactive) =====
def select_device():
    print("Select device for training:")
    print("1. CPU")
    print("2. CUDA (GPU, if available)")
    print("3. MPS (Apple Silicon)")
    choice = input("Enter choice (1/2/3): ").strip()
    if choice == "2" and torch.cuda.is_available():
        return torch.device("cuda")
    elif choice == "3" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

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

# ===== Training Loop =====
def train_model(model, train_loader, val_loader, device, epochs=3, lr=2e-5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train LLaVA-NeXT Classifier for Disaster Events")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Path to training data")
    parser.add_argument("--val_dir", type=str, default="data/test", help="Path to validation data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    
    device = select_device()
    print(f"Using device: {device}")
    
    train_dataset = DisasterImageDataset(args.train_dir, transform=transform)
    val_dataset = DisasterImageDataset(args.val_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    num_classes = len(train_dataset.event_categories)
    classifier_model = LlavaNextClassifier(vision_encoder, num_classes, device)
    classifier_model.to(device)
    
    train_model(classifier_model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)
    
    # Save the fine-tuned model
    torch.save(classifier_model.state_dict(), "./models/llava_next_classifier.pth")
    print("Saved fine-tuned model as llava_next_classifier.pth")
    
    # Evaluate on training and validation sets
    train_loss, train_acc = compute_accuracy(classifier_model, train_loader, device)
    val_loss, val_acc = compute_accuracy(classifier_model, val_loader, device)
    print(f"\nFinal Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
    print(f"Final Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
