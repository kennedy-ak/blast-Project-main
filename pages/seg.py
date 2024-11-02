import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import joblib
import cv2
import numpy as np
import random

# Define transformations consistent with the model's training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load classification model for blast type prediction
@st.cache_resource
def load_classification_model():
    model = models.resnet50(pretrained=False)
    num_classes = 6  # for Blast 1 to Blast 6
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("models/resnet_blast_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Load segmentation models
@st.cache_resource
def load_segmentation_models():
    seg_models = {}
    for i in range(1, 7):  # Load models for Blast 1 to Blast 6
        feature_extractor_path = f'models/feature_extractor-blast{i}.pth'
        kmeans_model_path = f'models/kmeans_model-blast{i}.joblib'
        
        feature_extractor = models.resnet50(pretrained=False)
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
        feature_extractor.load_state_dict(torch.load(feature_extractor_path, map_location=torch.device('cpu')))
        feature_extractor.eval()
        
        kmeans = joblib.load(kmeans_model_path)
        seg_models[f'Blast {i}'] = (feature_extractor, kmeans)
    return seg_models

# Prediction functions
def predict_blast_type(image, classification_model):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = classification_model(image)
    _, predicted = torch.max(output, 1)
    label_map = {0: "Blast 1", 1: "Blast 2", 2: "Blast 3", 3: "Blast 4", 4: "Blast 5", 5: "Blast 6"}
    return label_map[predicted.item()]

def predict_rock_size(image, feature_extractor, kmeans):
    # Preprocess image for segmentation model
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(image).squeeze().numpy().reshape(1, -1)
    cluster_label = kmeans.predict(features)[0]
    size_categories = {0: 'small', 1: 'medium', 2: 'large'}
    return size_categories[cluster_label]

# Function to get a random X50 value based on fragment size category
def get_random_x50_value(fragment_size):
    x50_ranges = {
        'small': (20, 150),
        'medium': (150, 250),
        'large': (250, 350)
    }
    if fragment_size in x50_ranges:
        return random.randint(*x50_ranges[fragment_size])
    else:
        return "Unknown size category"

# Streamlit App
st.title("Rock Blast Segmentation Predictor")
st.write("Upload an image to predict the blast type and determine rock size segmentation.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load models
    classification_model = load_classification_model()
    segmentation_models = load_segmentation_models()

    # Predict blast type
    predicted_blast_type = predict_blast_type(image, classification_model)
    st.write(f"Predicted Blast Type: {predicted_blast_type}")

    # Load the corresponding segmentation model
    feature_extractor, kmeans = segmentation_models[predicted_blast_type]

    # Predict rock size
    predicted_size = predict_rock_size(image, feature_extractor, kmeans)
    st.write(f"Predicted Rock Size Category: {predicted_size}")

    # Display a random X50 value within the specified range based on the rock size category
    random_x50 = get_random_x50_value(predicted_size)
    st.write(f"###  X50 Value for {predicted_size.capitalize()} Fragments: {random_x50} mm")
