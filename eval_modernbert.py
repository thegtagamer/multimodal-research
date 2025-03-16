import os
import torch
from torch.utils.data import Dataset, DataLoader  # ✅ Import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# **Disable FlashAttention if needed**
os.environ["USE_FLASH_ATTENTION"] = "0"

# **Select Device**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **Dataset Class (Same as Training)**
class DisasterTextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.texts = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        event_categories = {event: idx for idx, event in enumerate(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, event))}

        for event, label in event_categories.items():
            text_dir = os.path.join(data_path, event, "texts")
            
            if os.path.exists(text_dir):
                for text_file in os.listdir(text_dir):
                    if text_file.endswith(".txt"):
                        with open(os.path.join(text_dir, text_file), "r", encoding="utf-8") as file:
                            text_content = file.read().strip()
                            self.texts.append(text_content)
                            self.labels.append(label)

        print(f"Total test samples loaded: {len(self.texts)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# **Load Tokenizer & Dataset**
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

test_directory = "/scratch/adey6/multimodal-research/split_dataset/test"
test_dataset = DisasterTextDataset(test_directory, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# **Load the Best Model Checkpoint**
num_classes = len(set(test_dataset.labels))
model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-large", num_labels=num_classes)
model.load_state_dict(torch.load("best_modernbert_text_classifier.pth", map_location=device))  # ✅ Load best trained model
model.to(device)
model.eval()

# **Evaluate Model**
all_labels = []
all_preds = []
total_loss = 0
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# **Compute Metrics**
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")

print(f"\nValidation Loss: {total_loss / len(test_dataloader):.4f}")
print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")

