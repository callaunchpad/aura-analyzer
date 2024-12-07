import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torchvision.models import resnet18
from Pylette import extract_colors

# Dataset Class
class ColorDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Model Class
class ResNetColorAnalysis(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetColorAnalysis, self).__init__()
        self.base_model = resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Data Loader and Splitter
def load_and_split_data(data_dir, test_size=0.2, random_state=42):
    image_paths = []
    labels = []
    class_labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_labels)}

    for class_name in class_labels:
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if os.path.isfile(file_path):
                image_paths.append(file_path)
                labels.append(class_to_idx[class_name])

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return train_paths, test_paths, train_labels, test_labels, class_labels

# Training Function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu', save_path='../../color-analysis/trained_model.pth'):
    """Train the model and save it locally after training."""
    # Ensure the directory for the model save path exists
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Save the model after training
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


# Testing Function
def test_model(model, test_loader, device='cpu'):
    """
    Evaluate the model on the test set and return accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()  # Define loss for consistency with training

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total * 100
    return test_loss, test_accuracy

# Prediction Function
def predict_image(model, image_path, class_labels, transform, device='cpu'):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        return class_labels[pred.item()]

# Load Pretrained Model
def load_pretrained_model(model, save_path='../../color-analysis/trained_model.pth', device='cpu'):
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        print(f"Loaded pretrained model from {save_path}")
    else:
        print("No pretrained model found. Training from scratch.")
    return model

# Main Function
def main():
    parser = argparse.ArgumentParser(description="ResNet-18 Color Analysis")
    parser.add_argument("--image_path", "-i", help="Path to image for prediction", required=True)
    args = parser.parse_args()

    data_dir = "../../color-analysis/processed_images"
    num_classes = 4
    batch_size = 32
    num_epochs = 3
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = '../../color-analysis/trained_model.pth'  
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading and splitting data...")
    train_paths, test_paths, train_labels, test_labels, class_labels = load_and_split_data(data_dir)

    train_dataset = ColorDataset(train_paths, train_labels, transform=transform)
    test_dataset = ColorDataset(test_paths, test_labels, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ResNetColorAnalysis(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = load_pretrained_model(model, save_path, device)
    if not os.path.exists(save_path):
        print("Training the model...")
        train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device, save_path=save_path)

    print("Testing the model...")
    test_loss, test_accuracy = test_model(model, test_loader, device=device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    print(f"Predicting season for image: {args.image_path}")
    predicted_season = predict_image(model, args.image_path, class_labels, transform, device=device)
    if predicted_season:
        print(f"Predicted Season: {predicted_season}")
        palette = extract_colors(image=f'../../color-analysis/season-palettes/{predicted_season}.jpg', palette_size=48)
        palette.display(save_to_file=True, filename="../output-imgs/your-palette.png")
    else:
        print(f"Palette file for {predicted_season} not found.")
    

if __name__ == "__main__":
    main()
