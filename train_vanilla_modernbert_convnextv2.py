import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
from PIL import Image
from collections import Counter

# ✅ Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Dataset Class
class MultimodalDataset(Dataset):
    def __init__(self, data_path, tokenizer, image_processor, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        event_categories = {event: idx for idx, event in enumerate(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, event))}
        
        for event, label in event_categories.items():
            text_dir = os.path.join(data_path, event, "texts")
            image_dir = os.path.join(data_path, event, "images")

            if os.path.exists(text_dir) and os.path.exists(image_dir):
                text_files = {f.rsplit('.', 1)[0]: f for f in os.listdir(text_dir) if f.endswith(".txt")}
                image_files = {f.rsplit('.', 1)[0]: f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))}
                
                common_files = set(text_files.keys()) & set(image_files.keys())

                for file_name in common_files:
                    text_path = os.path.join(text_dir, text_files[file_name])
                    image_path = os.path.join(image_dir, image_files[file_name])
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
        image_inputs = self.image_processor(image, return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# ✅ Load Tokenizer & Processor
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
image_processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-large-22k-384")

train_directory = "/scratch/adey6/multimodal-research/split_dataset/train"
test_directory = "/scratch/adey6/multimodal-research/split_dataset/test"

train_dataset = MultimodalDataset(train_directory, tokenizer, image_processor)
test_dataset = MultimodalDataset(test_directory, tokenizer, image_processor)

# ✅ Class Balancing
class_counts = Counter([label for _, _, label in train_dataset.data])
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for _, _, label in train_dataset.data]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ✅ Model with Cross-Attention Fusion
class AdvancedMultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(AdvancedMultimodalModel, self).__init__()

        self.text_model = AutoModel.from_pretrained("answerdotai/ModernBERT-large")
        self.image_model = AutoModelForImageClassification.from_pretrained("facebook/convnextv2-large-22k-384", num_labels=num_classes, ignore_mismatched_sizes=True)

        self.text_fc = nn.Linear(self.text_model.config.hidden_size, 1024)
        self.image_fc = nn.Linear(1536, 1024)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.cross_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(1024)

        self.output_fc = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.relu(self.text_fc(text_outputs.last_hidden_state[:, 0, :]))

        image_outputs = self.image_model(pixel_values, output_hidden_states=True)
        image_features = image_outputs.hidden_states[-1]
        image_features = self.global_avg_pool(image_features).squeeze(-1).squeeze(-1)
        image_features = self.relu(self.image_fc(image_features))

        fusion_output, _ = self.cross_attn(text_features.unsqueeze(1), image_features.unsqueeze(1), image_features.unsqueeze(1))
        fusion_output = self.norm(fusion_output.squeeze(1))
        fusion_output = self.dropout(fusion_output)

        return self.output_fc(fusion_output)

# ✅ Training Setup
num_classes = len(set(train_dataset.data))
model = AdvancedMultimodalModel(num_classes).to(device)

optimizer = optim.AdamW([
    {"params": model.text_model.parameters(), "lr": 2e-5},
    {"params": model.image_model.parameters(), "lr": 2e-5},
    {"params": model.cross_attn.parameters(), "lr": 5e-4},  # Higher LR for fusion
], weight_decay=0.01)

criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# ✅ Train Model
def train_model(model, train_dataloader, optimizer, criterion, scheduler, epochs=10):
    best_val_loss = float("inf")
    patience = 3
    counter = 0
    checkpoint_path = "best_multimodal_model.pth"

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            input_ids, attention_mask, pixel_values, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["pixel_values"].to(device),
                batch["label"].to(device),
            )

            outputs = model(input_ids, attention_mask, pixel_values)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1} | Training Loss: {total_train_loss / len(train_dataloader):.4f}")

train_model(model, train_dataloader, optimizer, criterion, scheduler, epochs=10)

