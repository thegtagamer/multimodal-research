import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms, models
from PIL import Image, ImageFile
from tqdm import tqdm
from collections import Counter

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ✅ Select Device
def select_device():
    print("Select device for training:")
    print("1. CPU")
    print("2. CUDA (GPU, if available)")
    print("3. MPS (Apple Silicon, if available)")
    choice = input("Enter choice (1/2/3): ").strip()
    if choice == "2" and torch.cuda.is_available():
        return torch.device("cuda")
    elif choice == "3" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# ✅ Multimodal Dataset Class
class MultimodalDataset(Dataset):
    def __init__(self, data_path, tokenizer, image_transform, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

        event_categories = {event: idx for idx, event in enumerate(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, event))}
        for event, label in event_categories.items():
            text_dir = os.path.join(data_path, event, "texts")
            image_dir = os.path.join(data_path, event, "images")

            if os.path.exists(text_dir) and os.path.exists(image_dir):
                text_files = {f.rsplit('.', 1)[0]: f for f in os.listdir(text_dir) if f.endswith(".txt")}
                image_files = {f.rsplit('.', 1)[0]: f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))}

                common_files = set(text_files.keys()) & set(image_files.keys())
                for fname in common_files:
                    text_path = os.path.join(text_dir, text_files[fname])
                    image_path = os.path.join(image_dir, image_files[fname])
                    self.data.append((text_path, image_path, label))

        print(f"Total multimodal samples loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_path, image_path, label = self.data[idx]

        with open(text_path, "r", encoding="utf-8") as file:
            text = file.read().strip()
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pixel_values": image,
            "label": torch.tensor(label, dtype=torch.long)
        }

# ✅ Load Tokenizer & Image Transform
device = select_device()
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = "./data/train"
test_dir = "./data/test"

train_dataset = MultimodalDataset(train_dir, tokenizer, image_transform)
test_dataset = MultimodalDataset(test_dir, tokenizer, image_transform)

# ✅ Weighted Sampler
class_counts = Counter([label for _, _, label in train_dataset.data])
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for _, _, label in train_dataset.data]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_dataloader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ✅ EfficientNet + ModernBERT Multimodal Model
class EfficientNetBERTModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.text_model = AutoModel.from_pretrained("answerdotai/ModernBERT-large")
        self.text_fc = nn.Linear(self.text_model.config.hidden_size, 512)

        self.image_model = models.efficientnet_b3(pretrained=True)
        self.image_model.classifier = nn.Identity()
        self.image_fc = nn.Linear(1536, 512)

        self.fusion_fc = nn.Linear(1024, 512)
        self.output_fc = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.relu(self.text_fc(text_outputs.last_hidden_state[:, 0, :]))

        image_features = self.image_model(pixel_values)
        image_features = self.relu(self.image_fc(image_features))

        fused = torch.cat((text_features, image_features), dim=1)
        fused = self.relu(self.fusion_fc(fused))
        fused = self.dropout(fused)

        return self.output_fc(fused)

# ✅ Training Setup
num_classes = len(set([label for _, _, label in train_dataset.data]))
model = EfficientNetBERTModel(num_classes).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
class_weights_tensor = torch.tensor([class_weights[i] for i in range(num_classes)]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

best_val_loss = float("inf")
patience = 3
counter = 0

# ✅ Training Function
def train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, epochs=10):
    global best_val_loss, counter

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, pixel_values)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ✅ Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask, pixel_values)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ✅ Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), "./models/best_effnet_bert_model.pth")
            print("✅ Model saved (best so far).")
        else:
            counter += 1
            print(f"⏳ Early stopping patience: {counter}/{patience}")
        scheduler.step()

        # ✅ Early stopping
        if counter >= patience:
            print("🚨 Early stopping triggered.")
            break

# ✅ Run Training
train_model(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, epochs=10)

