import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from torchvision.models import efficientnet_b7
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

# **Dataset Class for Multimodal Fusion**
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
                
                # Find matching text-image pairs
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

        # Load text
        with open(text_path, "r", encoding="utf-8") as file:
            text = file.read().strip()

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }

# **Define Model**
class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalFusionModel, self).__init__()

        # Load BERT (Text Encoder)
        self.bert = BertModel.from_pretrained("bert-large-uncased")
        self.text_fc = nn.Linear(self.bert.config.hidden_size, 1024)

        # Load EfficientNet (Image Encoder)
        self.efficient_net = efficientnet_b7(pretrained=True)
        self.efficient_net.classifier = nn.Identity()  # Remove the classifier layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Ensure fixed output size
        self.image_fc = nn.Linear(2560, 1024)  # EfficientNet-B7 outputs 2560 features

        # **Fusion & Classification Layers**
        self.fusion_fc = nn.Linear(2048, 1024)
        self.output_fc = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, image):
        # **Text Features Extraction**
        text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_features = self.relu(self.text_fc(text_features))

        # **Image Features Extraction**
        image_features = self.efficient_net.features(image)  # Extract features
        image_features = self.adaptive_pool(image_features)  # Apply Adaptive Pooling
        image_features = torch.flatten(image_features, start_dim=1)  # Flatten
        image_features = self.relu(self.image_fc(image_features))  # Pass through FC layer

        # **Multimodal Fusion**
        multimodal_representation = torch.cat((image_features, text_features), dim=1)
        multimodal_representation = self.relu(self.fusion_fc(multimodal_representation))
        multimodal_representation = self.dropout(multimodal_representation)

        # **Final Classification**
        output = self.output_fc(multimodal_representation)
        return output

# **Training Script**
device = select_device()
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_directory = "./data/train"
test_directory = "./data/test"

train_dataset = MultimodalDisasterDataset(train_directory, tokenizer, transform=image_transforms)
test_dataset = MultimodalDisasterDataset(test_directory, tokenizer, transform=image_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

num_classes = len(set(train_dataset.data))
model = MultimodalFusionModel(num_classes).to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, epochs=3):
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        print(f"Epoch {epoch+1} | Training Loss: {total_train_loss / len(train_dataloader):.4f}")

train_model(model, train_dataloader, test_dataloader, optimizer, criterion, epochs=3)

torch.save(model.state_dict(), "./models/multimodal_fusion_model.pth")
