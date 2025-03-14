import torch
import torch.nn as nn
from torchvision.models import efficientnet_b7
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

# Select device
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

# Load Test Dataset
test_directory = "/scratch/adey6/multimodal-research/split_dataset/test"
test_dataset = DisasterImageDataset(test_directory, transform=image_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load Model (Ensure it Matches Training Structure)
num_classes = len(set(test_dataset.labels))
model = efficientnet_b7(pretrained=True)

# Modify Classifier Head to Match Training
model.classifier = nn.Sequential(
    nn.Dropout(0.3),  # Dropout for regularization
    nn.Linear(model.classifier[1].in_features, 1024),  # Fully connected 1024-neuron layer
    nn.ReLU(),  # Activation
    nn.Linear(1024, num_classes)  # Final classification layer
)

# Select Device
device = select_device()
print(f"Using device: {device}")
model.load_state_dict(torch.load("efficientnet_image_classifier.pth"))
model.to(device)
model.eval()

# Evaluate Model
def evaluate_model(model, dataloader, device):
    all_labels = []
    all_preds = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=True):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # Compute Validation Loss
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # # Confusion Matrix Plot
    # plt.figure(figsize=(8,6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # plt.title("Confusion Matrix")
    # plt.show()

# Run Evaluation
evaluate_model(model, test_dataloader, device)
