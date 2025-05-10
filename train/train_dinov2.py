import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
from PIL import Image, ImageFile
from collections import Counter

ImageFile.LOAD_TRUNCATED_IMAGES = True  # ✅ Prevents errors on incomplete images

# **Select Device**
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

# **Dataset Class for DINOv2**
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

        print(f"Total images loaded: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # **Process using DINOv2 processor (NO MANUAL NORMALIZATION)**
        inputs = self.processor(image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": label,
        }

# **Load Processor & Dataset**
device = select_device()
print(f"Using device: {device}")

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")  # ✅ Load DINOv2 processor

train_directory = "./data/train"
test_directory = "./data/test"

train_dataset = DisasterImageDataset(train_directory, processor)
test_dataset = DisasterImageDataset(test_directory, processor)

# **Handle Class Imbalance with Weighted Sampler**
class_counts = Counter(train_dataset.labels)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for label in train_dataset.labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_dataloader = DataLoader(train_dataset, batch_size=8, sampler=sampler)  # ✅ Use weighted sampler
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# **Load DINOv2 Model**
num_classes = len(set(train_dataset.labels))
model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-large", num_labels=num_classes).to(device)

# **Freeze Backbone for First Few Epochs**
for param in model.dinov2.parameters():
    param.requires_grad = False  # ✅ Freezing DINOv2 backbone initially

# **Define Optimizer, Weighted Loss & Learning Rate Scheduler**
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)  # ✅ Lower initial LR
class_weights_tensor = torch.tensor([class_weights[i] for i in range(num_classes)]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # ✅ Apply weighted loss

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # ✅ Reduce LR every 3 epochs

# **Early Stopping Parameters**
best_val_loss = float("inf")
patience = 3
counter = 0

# **Training Function**
def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, epochs=10):
    global best_val_loss, counter

    for epoch in range(epochs):
        if epoch == 3:  # ✅ Unfreeze backbone after 3 epochs
            for param in model.dinov2.parameters():
                param.requires_grad = True

        model.train()
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # **Validation Step**
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in test_dataloader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["label"].to(device)

                outputs = model(pixel_values, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_dataloader)

        print(f"Epoch {epoch+1} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        # **Model Checkpointing**
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), "./models/best_dinov2_image_classifier.pth")  # ✅ Save best model
            print("✅ Model saved as validation loss improved.")
        else:
            counter += 1
            print(f"⏳ Early stopping patience count: {counter}/{patience}")

        scheduler.step()

        # **Early Stopping**
        if counter >= patience:
            print("🚨 Early stopping triggered. Training stopped.")
            break

# **Train Model**
train_model(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, epochs=10)

