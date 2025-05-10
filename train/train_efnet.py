import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b7
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm

# Select device
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

# Define Dataset Class
class DisasterImageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

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

        if self.transform:
            image = self.transform(image)

        return image, label

# Define Image Transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Dataset
train_directory = "./data/train"
test_directory = "./data/test"

train_dataset = DisasterImageDataset(train_directory, transform=image_transforms)
test_dataset = DisasterImageDataset(test_directory, transform=image_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load Model
num_classes = len(set(train_dataset.labels))
model = efficientnet_b7(pretrained=True)

# Modify Classifier Head to Match Paper
model.classifier = nn.Sequential(
    nn.Dropout(0.3),  # Dropout for regularization
    nn.Linear(model.classifier[1].in_features, 1024),  # Fully connected 1024-neuron layer
    nn.ReLU(),  # Activation
    nn.Linear(1024, num_classes)  # Final classification layer
)

# Select Device
device = select_device()
print(f"Using device: {device}")
model.to(device)

# Define Optimizer & Loss Function
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# Training Function
def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, epochs=3):
    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        for images, labels in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", leave=True):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Evaluate on Validation Set
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_dataloader)

        print(f"Epoch {epoch+1} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

# Train Model
train_model(model, train_dataloader, test_dataloader, optimizer, criterion, epochs=3)

# Save Model
torch.save(model.state_dict(), "./models/efficientnet_image_classifier.pth")
