import sys
import os
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet18, ResNet18_Weights


import numpy as np
from Pylette import extract_colors
import argparse

# ImageClassifier model for color season prediction
class ImageClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_labels = ["Bright Spring", "True Spring", "Light Spring", "Warm Autumn"]
        self.model = self._create_model()

    def _create_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self,datapath="../../combined-demo/processed_images"):
        train_set = tf.keras.utils.image_dataset_from_directory(datapath, image_size=(224, 224),batch_size=32,seed=123, validation_split=0.2,subset="training")
        val_set = tf.keras.utils.image_dataset_from_directory(datapath, image_size=(224, 224),batch_size=32,seed=123, validation_split=0.2,subset="validation")
        class_names = ['autumn','spring','summer','winter']

        ds_train_images = np.concatenate([x.numpy() for x, y in train_set])
        ds_train_labels = np.concatenate([y.numpy() for x, y in train_set])

        ds_test_images = np.concatenate([x.numpy() for x, y in val_set])
        ds_test_labels = np.concatenate([y.numpy() for x, y in val_set])

        self.model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        self.model.fit(ds_train_images, ds_train_labels)
        test_loss, test_acc = self.model.evaluate(ds_test_images, ds_test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)

    def predict_season(self, image_path):
        img = load_img(image_path, target_size=self.input_shape[:2])
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        self.train()
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        season = self.class_labels[predicted_class]
        return season    
    

# Scrape images for training dataset
def scrape_and_download_images(target_dir="season-palettes"):
    url = "https://www.spicemarketcolour.com.au/celebrities"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    os.makedirs(target_dir, exist_ok=True)

    for section in soup.find_all("div", class_="season-section"):
        season_name = section.find("h2").text.strip().replace(" ", "_")
        season_dir = os.path.join(target_dir, season_name)
        os.makedirs(season_dir, exist_ok=True)

        for img_tag in section.find_all("img"):
            img_url = img_tag['src']
            img_name = os.path.basename(img_url)
            img_path = os.path.join(season_dir, img_name)
            try:
                img_data = requests.get(img_url).content
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                print(f"Downloaded {img_name} to {season_dir}")
            except Exception as e:
                print(f"Failed to download {img_url}: {e}")


# Display the color palette for the predicted season
def display_season_palette(season):
    palette_file = {
        "Bright Spring": '../../color-analysis/season-palettes/warm-spring.JPG',
        "True Spring": '../../color-analysis/season-palettes/warm-spring.JPG',
        "Light Spring": '../../color-analysis/season-palettes/light-spring.JPG',
        "Warm Autumn": '../../color-analysis/season-palettes/warm-autumn.JPG',
        "Soft Autumn": '../../color-analysis/season-palettes/soft-autumn.JPG',
        "Deep Autumn": '../../color-analysis/season-palettes/deep-autumn.JPG',
        "Clear Winter": '../../color-analysis/season-palettes/clear-winter.JPG',
        "Cool Winter": '../../color-analysis/season-palettes/cool-winter.JPG',
        "Deep Winter": '../../color-analysis/season-palettes/deep-winter.JPG',
        "Light Summer": '../../color-analysis/season-palettes/light-summer.JPG',
        "Cool Summer": '../../color-analysis/season-palettes/cool-summer.JPG',
        "Soft Summer": '../../color-analysis/season-palettes/soft-summer.JPG'
    }

    if season in palette_file:
        palette = extract_colors(image=palette_file[season], palette_size=48)
        palette.display(save_to_file=True, filename="../output-imgs/your-palette")
    else:
        print(f"No palette found for the season: {season}")

# Main function to predict and display season
def main(image_path=None):
    classifier = ImageClassifier(input_shape=(224, 224, 3), num_classes=4)
    
    if image_path:
        season = classifier.predict_season(image_path)
        print(f"Your color season is: {season}")
        display_season_palette(season)
    else:
        print("Error: No image path provided for analysis.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict color season and display palette")
    parser.add_argument("--image_path", "-i", help="Input image filename", required=True)
    args = parser.parse_args()

    # Run the main function
    main(image_path=args.image_path)
