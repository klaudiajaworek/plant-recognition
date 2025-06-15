import json
import timm
import torch
import folium
import pandas as pd
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

species_dir = "species_organs"
NUM_CLASSES = 65#len(pd.read_csv("data/plant_info.csv"))
with open("data/label_mapping.json", "r") as f:
    LABEL_MAPPING = json.load(f)


def get_plant_info(idx):
    df = pd.read_csv("data/plant_info.csv")
    return df[df["id"] == idx].iloc[0].to_dict()

def load_plant_info():
    df = pd.read_csv("data/plant_info.csv")
    return df

plant_df = load_plant_info()

# Utility functions

def get_plant_info_by_id(idx):
    return plant_df[plant_df["id"] == idx].iloc[0].to_dict()


def get_plant_info_by_latin(name):
    row = plant_df[plant_df["latin_name"] == name].iloc[0]
    return row.to_dict()


def safe(val):
    return "No data" if pd.isna(val) or val == "" else val



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
    model_eff.eval()
    outputs = model_eff(img_trans)
    predicted_class = torch.max(outputs, 1).indices.item()
    plant_idx = LABEL_MAPPING[str(predicted_class)]
    plant_info = get_plant_info(int(plant_idx))
    return plant_info

st.set_page_config(page_title="Plant Explorer", page_icon="🌱")


st.markdown(
    """
    <style>
    /*–––– page gradient ––––*/
    .stApp {
      background: linear-gradient(to bottom right, #a5d6a7, #17331a);
    }

    /*–––– hide the Streamlit toolbar and push content down ––––*/
    [data-testid="stToolbar"] {
      visibility: hidden;
    }

    /*–––– push the whole page down ––––*/
    div[class*="block-container"] {
      padding-top: 5rem !important;    /* was 2rem, bumped up to 5rem */
    }

    /*–––– or (instead) only push the H1 lower ––––*/
    [data-testid="stMarkdownContainer"] h1 {
      margin-top: 6rem !important;
    }

    /*–––– global green text ––––*/
    .stApp, 
    .stApp * {
      color: 
#d3f2d6
 !important;
    }
    ::placeholder {
      color: 
#a5d6a7
 !important;
      opacity: 1 !important;
    }

    /*–––– transparent “boxes” ––––*/
    div[class*="block-container"],
    [data-testid="stMarkdownContainer"] > div,
    [data-testid="stFileUploader"] > div {
      background-color: transparent !important;
      box-shadow: none !important;
      padding: 0 !important;
    }

    /*–––– lightly tinted inputs ––––*/
    [data-testid="stTextInput"] input,
    [data-testid="stFileUploader"] > div {
      background-color: rgba(200,230,201,0.8) !important;
      border-radius: 8px !important;
    }

    /*––– style the drop‐zone wrapper –––*/
    [data-testid="stFileUploader"] > div {
      /* pick up your green tint or even the gradient */
      background: rgba(200,230,201,0.8) !important;
      /* or, to use your page gradient:
         background: linear-gradient(to bottom right, #e8f5e9, #a5d6a7) !important;
      */
      border: 2px dashed #2e7d32 !important;
      border-radius: 8px !important;
      padding: 1rem !important;
      box-shadow: none !important;
    }

    /*––– remove the darker inner box so only your tint shows –––*/
    [data-testid="stFileUploader"] > div > div {
      background-color: transparent !important;
    }

    /*––– restyle the “Browse files” button –––*/
    [data-testid="stFileUploader"] button {
      background: rgba(200,230,201,0.8) !important;
      color: #2e7d32 !important;
      border: 1px solid #2e7d32 !important;
      border-radius: 4px !important;
      padding: 0.5em 1em !important;
    }

    
    </style>
    """,
    unsafe_allow_html=True,
)




def display_plant_details(result: dict):
    st.subheader("Prediction Result" if mode == "Predict" else "Species Details")
    st.markdown(f"**Latin Name:** {result['latin_name']}")
    # Robust handling of English names
    raw_eng = result.get("english_names", "")
    if not isinstance(raw_eng, str):
        raw_eng = ""
    names = [n.strip().title() for n in raw_eng.split(";") if n.strip()]
    eng_str = ", ".join(names) if names else ""
    st.markdown(f"**English Name(s):** {safe(eng_str)}")
    st.markdown(f"**Polish Name:** {safe(result.get('polish_names'))}")
    st.markdown(f"**Description:** {safe(result.get('description'))}")
    st.markdown(f"**Wikipedia link:** {safe(result.get('wiki_link'))}")


    # Example photos
    st.subheader("Example Photos")
    cols = st.columns(3)
    base = os.path.join(species_dir, str(result['id']))
    for organ, col in zip(["leaf", "flower", "fruit"], cols):
        with col:
            patterns = [f"*{organ}*.[jJ][pP][gG]", f"*{organ}*.[pP][nN][gG]"]
            files = []
            for pat in patterns:
                files = glob.glob(os.path.join(base, pat))
                if files: break
            if files:
                img = Image.open(files[0])
                st.image(img, caption=organ.title(), use_container_width=True)
            else:
                st.write(f"No {organ.title()} Image")

    # Distribution map
    st.subheader("Geographic Distribution")
    coords = [tuple(map(float, s.split('|'))) for s in result.get('lat_lon_coords','').split(';') if s]
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
    cluster = MarkerCluster().add_to(m)
    for lat, lon in coords:
        folium.CircleMarker(
            location=[lat, lon], radius=4, fill=True,
            fill_opacity=0.7, color='green', fill_color='green', weight=0
        ).add_to(cluster)
    st_folium(m, width="100%", height=300)

# Page Config & Styles
# Sidebar navigation
mode = st.sidebar.radio("Choose mode:", ["Predict", "Explore Species"]
)

if mode == "Predict":
    st.title("Plant Recognition App 🚀")
    st.write("Upload a plant image to get a prediction.")
    file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    if file:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        with st.spinner("Identifying plant..."):
            res = predict(img)
        display_plant_details(res)

else:
    st.title("Explore Species 🌿")
    # Alphabetical list of Latin names
    latin_list = sorted(plant_df['latin_name'].tolist())
    choice = st.selectbox("Select a species by Latin name:", latin_list)
    if choice:
        info = get_plant_info_by_latin(choice)
        display_plant_details(info)

