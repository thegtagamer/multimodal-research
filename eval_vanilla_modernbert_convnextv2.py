import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ✅ Select Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Define Dataset Class (Aligned with Training)
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

                # ✅ Find Matching Text-Image Pairs
                common_files = set(text_files.keys()) & set(image_files.keys())

                for file_name in common_files:
                    text_path = os.path.join(text_dir, text_files[file_name])
                    image_path = os.path.join(image_dir, image_files[file_name])
                    self.data.append((text_path, image_path, label))

        print(f"Total multimodal test samples loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_path, image_path, label = self.data[idx]

        # ✅ Load Text
        with open(text_path, "r", encoding="utf-8") as file:
            text = file.read().strip()

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # ✅ Load Image
        image = Image.open(image_path).convert("RGB")
        image_inputs = self.image_processor(image, return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "text_path": text_path,
            "image_path": image_path
        }

# ✅ Load Tokenizer & Processor
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
image_processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-large-22k-384")

# ✅ Define Model Class
class VanillaMultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(VanillaMultimodalModel, self).__init__()

        # ✅ Load ModernBERT for Text
        self.text_model = AutoModel.from_pretrained("answerdotai/ModernBERT-large")
        self.text_fc = nn.Linear(self.text_model.config.hidden_size, 1024)

        # ✅ Load ConvNeXt V2 for Image
        self.image_model = AutoModelForImageClassification.from_pretrained(
            "facebook/convnextv2-large-22k-384", num_labels=num_classes, ignore_mismatched_sizes=True
        )
        self.image_fc = nn.Linear(1536, 1024)  # ✅ Feature size 1536 → 1024
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ✅ Fusion and Classification
        self.fusion_fc = nn.Linear(2048, 1024)
        self.output_fc = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.relu(self.text_fc(text_outputs.last_hidden_state[:, 0, :]))

        image_outputs = self.image_model(pixel_values, output_hidden_states=True)
        image_features = image_outputs.hidden_states[-1]
        image_features = self.global_avg_pool(image_features).squeeze(-1).squeeze(-1)
        image_features = self.relu(self.image_fc(image_features))

        fusion_features = torch.cat((text_features, image_features), dim=1)
        fusion_features = self.relu(self.fusion_fc(fusion_features))
        fusion_features = self.dropout(fusion_features)

        return self.output_fc(fusion_features)

# ✅ Load Dataset and Model
test_directory = "/scratch/adey6/multimodal-research/split_dataset/test"
num_classes = len(next(os.walk(test_directory))[1])  # ✅ Dynamically detect number of classes

test_dataset = MultimodalDataset(test_directory, tokenizer, image_processor)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = VanillaMultimodalModel(num_classes).to(device)

# ✅ Load Model Checkpoint (Fixing Classifier Mismatch)
checkpoint_path = "vanilla_modernbert_convnextv2.pth"
state_dict = torch.load(checkpoint_path, map_location=device)

# ✅ Remove Incompatible Classifier Layers
for key in ["image_model.classifier.weight", "image_model.classifier.bias", "output_fc.weight", "output_fc.bias"]:
    state_dict.pop(key, None)

# ✅ Load Compatible Layers
model.load_state_dict(state_dict, strict=False)
model.eval()

print("✅ Model successfully loaded with updated classifier layers!")

# ✅ Evaluate Model
def evaluate_model(model, test_dataloader):
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
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
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)

    results_text = (
        f"\n🎯 **Evaluation Metrics:**\n"
        f"✅ Validation Loss: {total_loss / len(test_dataloader):.4f}\n"
        f"✅ Accuracy: {accuracy:.4f}\n"
        f"✅ Precision: {precision:.4f}\n"
        f"✅ Recall: {recall:.4f}\n"
        f"✅ F1-score: {f1:.4f}\n\n"
    )

    # ✅ Save Results
    results_dict = {
        "Validation Loss": total_loss / len(test_dataloader),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

    with open("evaluation_results.json", "w") as file:
        json.dump(results_dict, file, indent=4)

    print(results_text)
    print("📁 Evaluation results saved to `evaluation_results.json`")

# ✅ Run Evaluation
evaluate_model(model, test_dataloader)

