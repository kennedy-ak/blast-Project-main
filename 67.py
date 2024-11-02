import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import math

# Load blast parameters from the uploaded CSV file
blast_parameters = pd.read_csv("paramm.csv")

# Define the transformation (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load the trained ResNet model
@st.cache_resource
def load_model():
    model = models.resnet50()
    num_classes = 6  # Blast 1 to Blast 6
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("models/resnet_blast_model.pth", map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model

# Function to make predictions
def predict(image, model):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    label_map = {0: "Blast 1", 1: "Blast 2", 2: "Blast 3", 3: "Blast 4", 4: "Blast 5", 5: "Blast 6"}
    return label_map[predicted.item()]

# Function to get selected blast parameters from prediction
def get_selected_parameters(predicted_class):
    parameters = blast_parameters[blast_parameters["Blast_Number"] == predicted_class]
    if not parameters.empty:
        selected_parameters = parameters[[
            "Bench Height", "Burden", "Spacing", "Hole Diameter", 
            "Rock factor", "Standard deviation of drilling accuracy", 
            "Quantity of explosive per hole", "Powder factor", "Charge length"
        ]]
        return selected_parameters
    return None

# Function to calculate X50 using the Kuznetsov equation
def calculate_X50(A, K, Q, RWS):
    # Kuznetsov equation: X_m = AK^(-0.8)Q^(1/6)(115/RWS)^(19/20)
    X_m = A * (K ** -0.8) * (Q ** (1/6)) * ((115 / RWS) ** (19 / 20))
    return X_m

# Function to calculate Uniformity Index (N)
def calculate_N(B, S, d, W, L, H):
    # Uniformity index equation without the abs(BCL - CCL/L) term
    n = (2.2 - (14 * B) / d) * math.sqrt((1 + S / B) / 2) * (1 - W / B) * (L / H)
    return n

# Streamlit UI
st.title("Blast Classification and Parameter Calculation App")
st.write("Upload an image to classify the blast type (Blast 1 to Blast 6) and calculate X50 and Uniformity Index (N).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the model and make predictions
    model = load_model()
    prediction = predict(image, model)

    # Display the prediction
    st.write(f"Predicted Class: **{prediction}**")

    # Get and display the corresponding blast parameters
    selected_parameters = get_selected_parameters(prediction)
    if selected_parameters is not None:
        st.write("### Selected Blast Parameters:")
        st.dataframe(selected_parameters)

        # Extract the necessary parameters for X50 calculation
        A = selected_parameters["Rock factor"].values[0]
        K = selected_parameters["Powder factor"].values[0]
        Q = selected_parameters["Quantity of explosive per hole"].values[0]
        RWS = 85  # Example value for RWS

        # Extract additional parameters for N calculation
        B = selected_parameters["Burden"].values[0]
        S = selected_parameters["Spacing"].values[0]
        d = selected_parameters["Hole Diameter"].values[0]
        W = selected_parameters["Standard deviation of drilling accuracy"].values[0]
        L = selected_parameters["Charge length"].values[0]
        H = selected_parameters["Bench Height"].values[0]

        # Initial calculation of X50 and N
        X50_initial = calculate_X50(A, K, Q, RWS)
        N_initial = calculate_N(B, S, d, W, L, H)

        # Display the initial calculated X50 and N
        st.write(f"### Initial Calculated X50 (Mean Particle Size): {X50_initial:.2f} cm")
        st.write(f"### Initial Calculated N (Uniformity Index): {N_initial:.2f}")

        st.write("### Modify Parameters to See How They Affect the Values:")
        
        # User input for modifying parameters
        A_mod = st.number_input("Rock factor (A)", value=float(A))
        K_mod = st.number_input("Powder factor (K)", value=float(K))
        Q_mod = st.number_input("Quantity of explosive per hole (Q)", value=float(Q))
        RWS_mod = st.number_input("Relative weight strength (RWS)", value=float(RWS))
        B_mod = st.number_input("Burden (B)", value=float(B))
        S_mod = st.number_input("Spacing (S)", value=float(S))
        d_mod = st.number_input("Hole Diameter (d)", value=float(d))
        W_mod = st.number_input("Standard deviation of drilling accuracy (W)", value=float(W))
        L_mod = st.number_input("Charge length (L)", value=float(L))
        H_mod = st.number_input("Bench Height (H)", value=float(H))

        # Add a calculate button
        if st.button('Calculate'):
            # Recalculate X50 and N with modified values
            X50_modified = calculate_X50(A_mod, K_mod, Q_mod, RWS_mod)
            N_modified = calculate_N(B_mod, S_mod, d_mod, W_mod, L_mod, H_mod)

            # Display the modified calculated X50 and N
            st.write(f"### Modified Calculated X50 (Mean Particle Size): {X50_modified:.2f} cm")
            st.write(f"### Modified Calculated N (Uniformity Index): {N_modified:.2f}")
    else:
        st.write("No parameters found for the predicted class.")
