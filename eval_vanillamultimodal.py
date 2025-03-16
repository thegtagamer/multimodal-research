import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b7
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allows loading of truncated images

import os

# **Select Device**
def select_device():
    print("Select device for evaluation:")
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

# **Dataset Class**
class MultimodalDisasterDataset(Dataset):
    def __init__(self, data_path, tokenizer, transform=None, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.transform = transform
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

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }

# **Define Model Independently**
class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalFusionModel, self).__init__()

        self.bert = BertModel.from_pretrained("bert-large-uncased")
        self.text_fc = nn.Linear(self.bert.config.hidden_size, 1024)

        self.efficient_net = efficientnet_b7(pretrained=True)
        self.efficient_net.classifier = nn.Identity()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_fc = nn.Linear(2560, 1024)

        self.fusion_fc = nn.Linear(2048, 1024)
        self.output_fc = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, image):
        text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_features = self.relu(self.text_fc(text_features))

        image_features = self.efficient_net.features(image)
        image_features = self.adaptive_pool(image_features)
        image_features = torch.flatten(image_features, start_dim=1)
        image_features = self.relu(self.image_fc(image_features))

        multimodal_representation = torch.cat((image_features, text_features), dim=1)
        multimodal_representation = self.relu(self.fusion_fc(multimodal_representation))
        multimodal_representation = self.dropout(multimodal_representation)

        output = self.output_fc(multimodal_representation)
        return output

# **Load Tokenizer & Dataset**
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_directory = "/scratch/adey6/multimodal-research/split_dataset/test"
test_dataset = MultimodalDisasterDataset(test_directory, tokenizer, transform=image_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# **Load Model Independently**
device = select_device()
print(f"Using device: {device}")

# ✅ **Fix: Load correct `num_classes` from checkpoint**
checkpoint = torch.load("multimodal_fusion_model.pth", map_location=device)
num_classes = checkpoint["output_fc.bias"].shape[0]

model = MultimodalFusionModel(num_classes)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# **Evaluate Model**
all_labels = []
all_preds = []
total_loss = 0
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")

print(f"Validation Loss: {total_loss / len(test_dataloader):.4f}")
print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")

#plt.figure(figsize=(8,6))
#sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues')
#plt.xlabel("Predicted Label")
#plt.ylabel("True Label")
#plt.title("Confusion Matrix")
#plt.show()

