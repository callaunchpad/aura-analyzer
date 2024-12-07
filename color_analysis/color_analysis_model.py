import os
import numpy as np
import argparse
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from Pylette import extract_colors


class ColorDataset(torch.utils.data.Dataset):
    """Defines the dataset with image paths and their corresponding labels."""
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


class ResNetColorAnalysis(nn.Module):
    """Defines the ResNet-based model for color abalysis classification"""
    def __init__(self, num_classes=4):
        super(ResNetColorAnalysis, self).__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


def load_and_split_data(data_dir, test_size=0.2, random_state=42):
    """Load image paths and labels, then split them into training and testing sets."""
    image_paths = []
    labels = []
    class_labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_labels)}

    # Collect image paths and labels from each class folder
    for class_name in class_labels:
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if os.path.isfile(file_path):
                image_paths.append(file_path)
                labels.append(class_to_idx[class_name])

    # Split the data into training and testing sets
    train_set, test_set, train_labels, test_labels = train_test_split(image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels)
    return train_set, test_set, train_labels, test_labels, class_labels


def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu', save_path='color_analysis/trained_model.pth'):
    """Train the model and save it locally after training."""
    # Ensure the directory for the model save path exists
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Model is set to training mode
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


def test_model(model, test_loader, device='cpu'):
    """Evaluate the model on the test set and return accuracy"""
    # Model is set to evaluation mode
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()  

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


def predict_image(model, image_path, class_labels, transform, device='cpu'):
    """Predicts the class of a single image"""
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        return class_labels[pred.item()]


def load_pretrained_model(model, save_path='color_analysis/trained_model.pth', device='cpu'):
    """Loads a previously trained model or starts from scratch if no model is found"""
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        model.eval()
    return model


def save_season_palette(predicted_season):
    palette = extract_colors(image=f'color_analysis/nonspecific-season-palettes/{predicted_season}-palette.jpg', palette_size=48)
    w, h = 48, 48
    img = Image.new("RGB", size=(w * palette.number_of_colors, h))
    arr = np.asarray(img).copy()
    for i in range(palette.number_of_colors):
        c = palette.colors[i]
        arr[:, i * h : (i + 1) * h, :] = c.rgb
    img = Image.fromarray(arr, "RGB")

    img.save(f"combined_demo/output-imgs/your-palette.jpg")


def main():
    # Get the image path for the prediction
    parser = argparse.ArgumentParser(description="ResNet-18 Color Analysis")
    parser.add_argument("--image_path", "-i", help="Path to image for prediction", required=True)
    args = parser.parse_args()

    # Parameters
    data_dir = "color_analysis/processed_images"
    num_classes = 4
    batch_size = 32
    num_epochs = 3
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = 'color_analysis/trained_model.pth'  

    # Transform the input image for consistency
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create training and testing sets
    train_set, test_set, train_labels, test_labels, class_labels = load_and_split_data(data_dir)
    train_dataset = ColorDataset(train_set, train_labels, transform=transform)
    test_dataset = ColorDataset(test_set, test_labels, transform=transform)

    # Create DataLoaders for batch processing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initalize, load, and evaluate the model's performance on the test set
    model = ResNetColorAnalysis(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = load_pretrained_model(model, save_path, device)
    if not os.path.exists(save_path):
        train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device, save_path=save_path)

    # test_loss, test_accuracy = test_model(model, test_loader, device=device)
    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Predict the season of the input image with the trained model
    predicted_season = predict_image(model, args.image_path, class_labels, transform, device=device)
    if predicted_season:
        print(f"your color season is {predicted_season}")
        save_season_palette(predicted_season)    
    else:
        print(f"your color season is null")
    

if __name__ == "__main__":
    main()
