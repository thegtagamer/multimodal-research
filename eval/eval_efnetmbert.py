import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms, models
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ✅ Select Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Multimodal Dataset Class (same as training)
class MultimodalDataset(torch.utils.data.Dataset):
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

        print(f"Total multimodal test samples loaded: {len(self.data)}")

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
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Define Model
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

# ✅ Load Data
test_directory = "./data/test"
num_classes = len(next(os.walk(test_directory))[1])
test_dataset = MultimodalDataset(test_directory, tokenizer, image_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ✅ Initialize Model
model = EfficientNetBERTModel(num_classes).to(device)
model.load_state_dict(torch.load("./models/best_effnet_bert_model.pth", map_location=device))
model.eval()

print("✅ Model loaded successfully!")

# ✅ Evaluate Model
all_preds, all_labels = [], []
criterion = nn.CrossEntropyLoss()
total_loss = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask, pixel_values)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

# ✅ Compute Metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
avg_loss = total_loss / len(test_loader)

results = {
    "Validation Loss": avg_loss,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1
}

# # ✅ Save Results
# with open("effnet_bert_evaluation_results.json", "w") as f:
#     json.dump(results, f, indent=4)

print("\n🎯 Evaluation Metrics:")
print(f"✅ Validation Loss: {avg_loss:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-score: {f1:.4f}")
# print("📁 Results saved to `effnet_bert_evaluation_results.json`")

