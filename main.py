import json
import timm
import torch
import folium
import pandas as pd
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
from streamlit_folium import st_folium

NUM_CLASSES = len(pd.read_csv("data/plant_info.csv"))
with open("data/label_mapping.json", "r") as f:
    LABEL_MAPPING = json.load(f)


def get_plant_info(idx):
    df = pd.read_csv("data/plant_info.csv")
    return df[df["id"] == idx].iloc[0].to_dict()


def create_efficientnet_b0(num_classes):
    model = timm.create_model(
        "efficientnet_b0", pretrained=True, num_classes=num_classes
    )
    return model


test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict(img):
    img_trans = test_transform(img.convert("RGB")).unsqueeze(0)
    model_eff = create_efficientnet_b0(NUM_CLASSES)
    model_eff.load_state_dict(
        torch.load("data/efficientnet_b0_plants.pth", map_location=torch.device("cpu"))
    )
    outputs = model_eff(img_trans)
    predicted_class = torch.max(outputs, 1).indices.item()
    plant_idx = LABEL_MAPPING[str(predicted_class)]
    plant_info = get_plant_info(int(plant_idx))
    return plant_info


# Title
st.title("Plant Recognition App")
st.write("Upload a plant image to get a prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    with st.spinner("Identifying plant..."):
        result = predict(image)

    # Show results
    st.subheader("Prediction Result")
    st.markdown(f"**English Name:** {result['english_names']}")
    st.markdown(f"**Polish Name:** {result['polish_names']}")
    st.markdown(f"**Description:** {result['description']}")

    # Display map
    st.subheader("Geographic Distribution")
    plant_map = folium.Map(location=[20, 0], zoom_start=2)
    for lat, lon in result["distribution"]:
        folium.Marker(location=[lat, lon], popup=result["english_name"]).add_to(
            plant_map
        )

    st_folium(plant_map, width=700, height=450)
