from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from typing import List, Optional
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
import json
import google.generativeai as genai
from PIL import Image
import shutil
genai.configure(api_key = 'AIzaSyAhSq07dLuxm4RnO1frxNgUBufydWpTTsw')
app = FastAPI()

# CORS middleware
from fastapi.middleware.cors import CORSMiddleware
origins = ["https://78lw0q6f-5173.inc1.devtunnels.ms/"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
labels = []
base_learning_rate = 0.0001
num_classes = 0
h5_file_path = "face-net-savedmodel.keras"  # Path to pre-trained model file

# Directories
UPLOAD_DIR = "uploaded_images"
IMAGES_DIR = "Images"
TEMP_UPLOAD_DIR = "temp_uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

def clear_directory(directory_path):
    """Clear the contents of a directory."""
    if os.path.exists(directory_path):
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def preprocess_image(image_path):
    """Preprocess the image for prediction."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)  # Normalize to [0, 1]
    return np.expand_dims(img_array, 0)

def predict_class(img_array, model, labels, confidence_threshold=0.4):
    """Predict the class of an image array."""
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    prediction_confidence = np.max(prediction)

    if prediction_confidence < confidence_threshold:
        return "Unknown"

    return labels[predicted_class]

@app.on_event("startup")
def load_pretrained_model():
    """Load the pretrained model and labels on startup."""
    global model, labels, num_classes
    clear_directory(UPLOAD_DIR)  # Clear old uploaded files

    if os.path.exists(h5_file_path):
        model = load_model(h5_file_path)
        labels_file_path = h5_file_path.replace(".keras", "_labels.json")
        if os.path.exists(labels_file_path):
            with open(labels_file_path, "r") as f:
                labels = json.load(f)
        num_classes = len(labels)
    else:
        labels = []
        base_model = MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
        base_model.trainable = True

        global_average_layer = GlobalAveragePooling2D()(base_model.output)
        dropout_layer = Dropout(0.2)(global_average_layer)
        prediction_layer = Dense(num_classes, activation="softmax")(dropout_layer)

        model = Model(inputs=base_model.input, outputs=prediction_layer)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

@app.post("/train")
def train_model(
    new_images: Optional[List[UploadFile]] = None,
    new_names: Optional[List[str]] = None
):
    """Train the model using images in the directories or new uploads."""
    global model, labels, num_classes

    # Validate input
    if new_images and not new_names:
        raise HTTPException(status_code=400, detail="Please provide names corresponding to the uploaded images.")
    if new_names and not new_images:
        raise HTTPException(status_code=400, detail="Please provide images corresponding to the given names.")
    if new_images and len(new_images) != len(new_names):
        raise HTTPException(status_code=400, detail="Number of images and names must match.")

    # Clear temporary upload directory
    for filename in os.listdir(TEMP_UPLOAD_DIR):
        file_path = os.path.join(TEMP_UPLOAD_DIR, filename)
        os.remove(file_path)

    # Save new uploaded images
    if new_images:
        for img, name in zip(new_images, new_names):
            file_path = os.path.join(TEMP_UPLOAD_DIR, f"{name}.jpg")
            with open(file_path, "wb") as buffer:
                buffer.write(img.file.read())

    # Load images and labels from directories
    images = []
    dataset_labels = []

    for directory in [IMAGES_DIR, TEMP_UPLOAD_DIR]:
        for filename in os.listdir(directory):
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0  # Normalize images
            images.append(img_array)
            dataset_labels.append(filename.split(".")[0])

    # Create label mapping
    labels = sorted(set(dataset_labels))
    label_map = {label: idx for idx, label in enumerate(labels)}
    encoded_labels = [label_map[label] for label in dataset_labels]
    num_classes = len(labels)

    # Prepare datasets
    images = np.array(images)
    encoded_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes=num_classes)

    # Define data augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2)
    ])

    def preprocess(img, label):
        img = data_augmentation(img)  # Apply data augmentation
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((images, encoded_labels))
    dataset = dataset.map(preprocess).shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)

    # Initialize or update model
    if model is None or num_classes != model.output_shape[-1]:
        base_model = MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
        base_model.trainable = True

        global_average_layer = GlobalAveragePooling2D()(base_model.output)
        dropout_layer = Dropout(0.2)(global_average_layer)
        prediction_layer = Dense(num_classes, activation="softmax")(dropout_layer)

        model = Model(inputs=base_model.input, outputs=prediction_layer)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    # Train the model
    model.fit(dataset, epochs=300)

    # Save the model and labels
    model.save(h5_file_path)
    labels_file_path = h5_file_path.replace(".keras", "_labels.json")
    with open(labels_file_path, "w") as f:
        json.dump(labels, f)

    return {"message": "Model trained successfully", "labels": labels}


@app.post("/predict")
def predict(image: UploadFile = File(...)):
    """Predict the class of an uploaded image."""
    if model is None or not labels:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")

    file_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(image.file.read())

    img_array = preprocess_image(file_path)

    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    confidence_threshold = 0.6
    if confidence < confidence_threshold:
        predicted_label = "Unknown"
    else:
        predicted_label = labels[predicted_class_idx]

    os.remove(file_path)

    return {
        "predicted_label": predicted_label,
        "confidence": float(confidence)
    }


@app.post("/describe-image")
def describe_image(image: UploadFile = File(...)):
    """Generate a description of an uploaded image using Generative AI."""
    file_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(image.file.read())

    img = Image.open(file_path)
    generative_model = genai.GenerativeModel('gemini-1.5-flash')
    response = generative_model.generate_content([
        img, "give simple description of the image. I want output in plain text."
    ])

    return {"description": response.text}
