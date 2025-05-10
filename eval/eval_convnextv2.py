import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True  # ✅ Prevents errors on incomplete images

# **Dataset Class for ConvNeXt V2 (Same as Training)**
class DisasterImageDataset(Dataset):
    def __init__(self, data_path, processor):
        self.image_paths = []
        self.labels = []
        self.processor = processor

        event_categories = {event: idx for idx, event in enumerate(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, event))}

        for event, label in event_categories.items():
            image_dir = os.path.join(data_path, event, "images")

            if os.path.exists(image_dir):
                for img_file in os.listdir(image_dir):
                    if img_file.endswith((".jpg", ".png", ".jpeg")):
                        self.image_paths.append(os.path.join(image_dir, img_file))
                        self.labels.append(label)

        print(f"Total test images loaded: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # **Process using ConvNeXt V2 processor**
        inputs = self.processor(image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": label,
        }

# **Load Processor & Dataset**
processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-large-22k-384")

test_directory = "./data/test"

test_dataset = DisasterImageDataset(test_directory, processor)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# **Load Model**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = len(set(test_dataset.labels))

# **Fix: Load Model with Correct Classifier Head**
model = AutoModelForImageClassification.from_pretrained(
    "facebook/convnextv2-large-22k-384",
    num_labels=num_classes,
    ignore_mismatched_sizes=True  # ✅ Prevents classifier mismatch errors
).to(device)

# **Load Best Model Checkpoint**
model.load_state_dict(torch.load("./models/best_convnextv2_image_classifier.pth", map_location=device))  
model.to(device)
model.eval()

# **Evaluate Model**
all_labels = []
all_preds = []
total_loss = 0
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        outputs = model(pixel_values)
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

