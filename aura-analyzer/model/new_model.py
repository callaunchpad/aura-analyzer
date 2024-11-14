import sys
import os
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
from PIL import Image
from io import BytesIO

class ImageClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_labels = ["Bright Spring", "True Spring", "Light Spring", "Warm Autumn"]
        self.model = self._create_model()

    def _create_model(self):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, train_dir, batch_size=32, epochs=25):
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        validation_generator = datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        self.model.fit(train_generator, validation_data=validation_generator, epochs=epochs)

    def convert_to_jpg(self, image_path):
        img = Image.open(image_path)
        if img.format != 'JPEG':
            img = img.convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            return buffer
        else:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            return buffer

    def predict_season(self, image_path):
        jpg_image_buffer = self.convert_to_jpg(image_path)
        img = load_img(jpg_image_buffer, target_size=self.input_shape[:2])
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        season = self.class_labels[predicted_class]
        return season, confidence

def scrape_and_download_images():
    url = "https://www.spicemarketcolour.com.au/celebrities"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    image_dir = "color_season_images"

    os.makedirs(image_dir, exist_ok=True)

    for section in soup.find_all("div", class_="season-section"):
        season_name = section.find("h2").text.strip().replace(" ", "_")
        season_dir = os.path.join(image_dir, season_name)
        os.makedirs(season_dir, exist_ok=True)

        for img_tag in section.find_all("img"):
            img_url = img_tag['src']
            img_name = os.path.basename(img_url)
            img_path = os.path.join(season_dir, img_name)
            img_data = requests.get(img_url).content
            with open(img_path, 'wb') as f:
                f.write(img_data)
            print(f"Downloaded {img_name} to {season_dir}")

if __name__ == "__main__":
    # Scrape and download images from the website
    scrape_and_download_images()

    # Specify the directory where images were downloaded for training
    train_data_dir = "color_season_images"

    # Initialize and train the model
    classifier = ImageClassifier(input_shape=(224, 224, 3), num_classes=4)
    classifier.train_model(train_data_dir, batch_size=32, epochs=25)

    # Example usage for prediction (replace with an actual image path if needed)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        season, confidence = classifier.predict_season(image_path)
        print(f"Predicted season: {season} with confidence {confidence:.2f}")
