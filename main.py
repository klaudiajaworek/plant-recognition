import json
import timm
import torch
import folium
import pandas as pd
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import requests
import geopandas as gpd
from shapely.geometry import Point, shape
from PIL import Image

species_dir = "species_organs"
NUM_CLASSES = 65 #len(pd.read_csv("data/plant_info.csv"))
with open("data/label_mapping.json", "r") as f:
    LABEL_MAPPING = json.load(f)


st.set_page_config(page_title="Plant Explorer", page_icon="🌱")


def get_plant_info(idx):
    df = pd.read_csv("data/plant_info.csv")
    return df[df["id"] == idx].iloc[0].to_dict()

@st.cache_data
def load_plant_info():
    df = pd.read_csv("data/plant_info.csv")
    return df

@st.cache_data
def load_world_geojson():
    # Fetch GeoJSON world boundaries
    url = (
        "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json"
    )
    return requests.get(url).json()

plant_df = load_plant_info()
world_geojson = load_world_geojson()

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
#e6f7e7
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

    # Species distribution heatmap
    st.subheader("Geographic Distribution")

    raw_coords = result.get('lat_lon_coords', '')
    coords = [tuple(map(float, s.split('|'))) for s in raw_coords.split(';') if s]

    obs_count = len(coords)
    st.markdown(f"**Species observations ({obs_count} observations)**")
    # Satellite basemap
    m = folium.Map(
        location=[20, 0],
        zoom_start=2,
        tiles='Esri.WorldImagery',
        attr='Esri'
    )
    # Heatmap overlay
    if coords:
        HeatMap(
            coords,
            radius=10,             # make each point a bit bigger
            blur=10,               # soften edges
            min_opacity=0.5,       # show even low-density areas
            max_zoom=6,
            gradient={
                0.0: 'blue',       # background/no data
                0.2: 'lime',       # low density
                0.4: 'yellow',     # medium density
                0.6: 'orange',     # moderately high
                0.9: 'red',        # high density
                1.0: 'darkred'     # peak density
            }
        ).add_to(m)
    # Layer control
    folium.LayerControl().add_to(m)
    st_folium(m, width="100%", height=400)

# Page Config & Styles
# Sidebar navigation
mode = st.sidebar.radio("Choose mode:", ["Predict", "Explore Species"]
)

if mode == "Predict":
    st.title("Plant Recognition App 🚀")
    st.write("Upload a plant image to get a prediction.")
    file = st.file_uploader("Choose an image... 🌱", type=["jpg","jpeg","png"])
    if file:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        with st.spinner("Identifying plant..."):
            res = predict(img)
        display_plant_details(res)

else:
    st.title("Explore Species 🌿")
    # session state init
    if 'show_map' not in st.session_state:
        st.session_state.show_map = False
    if 'country_choice' not in st.session_state:
        st.session_state.country_choice = 'All countries'
    if 'temp_country' not in st.session_state:
        st.session_state.temp_country = None

    # Map selection flow
    if st.session_state.show_map:
        st.subheader("Select a Country on Map")
        st.markdown("Double-click to select, then click the **Accept** button.")
        fmap = folium.Map(location=[20, 0], zoom_start=2, tiles='cartodbpositron')
        folium.GeoJson(
            world_geojson,
            name='countries',
            tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Country:']),
            highlight_function=lambda feat: {
                "fillColor": "#ffaa00",
                "weight": 1.5,
                "fillOpacity": 0.5,
            },
            style_function=lambda feat: {
                'fillColor': 'lime' if feat['properties']['name']==st.session_state.temp_country else 'transparent',
                'color': '#444','weight':0.5
            }
        ).add_to(fmap)
        clicked = st_folium(fmap, width="100%", height=500)
        latlng = clicked.get('last_clicked')
        if latlng:
            pt = Point(latlng['lng'], latlng['lat'])
            for feat in world_geojson.get('features', []):
                geom = shape(feat.get('geometry', {}))
                if geom.contains(pt):
                    st.session_state.temp_country = feat['properties'].get('name')
        if st.session_state.temp_country:
            st.success(f"You have selected {st.session_state.temp_country}")
            if st.button("Accept and return to explorer"):
                st.session_state.country_choice = st.session_state.temp_country
                st.session_state.temp_country = None
                st.session_state.show_map = False
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
                
        if st.button("Back to explorer without selecting"):
            st.session_state.temp_country = None
            st.session_state.show_map = False
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()

        st.stop()

    # Dropdown list
    plant_df['country_names'] = plant_df['country_names'].fillna('')
    all_countries = sorted(
        set(n.strip() for names in plant_df['country_names'].str.split(';') for n in (names or []) if n)
    )
    dropdown = ['All countries'] + all_countries
    try:
        initial_index = dropdown.index(st.session_state.country_choice)
    except ValueError:
        st.warning(f"'{st.session_state.country_choice}' is not available in dataset. Please select another country.")
        st.session_state.country_choice = 'All countries'
        initial_index = 0

    country = st.selectbox("Select a country:", dropdown, index=initial_index)
    st.session_state.country_choice = country

    # Button under dropdown
    if st.button("Select Country on Map"):
        st.session_state.show_map = True
        # use whichever rerun works for your Streamlit version:
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

    # Filter & list species
    mask = plant_df['country_names'].str.contains(rf"\b{st.session_state.country_choice}\b", na=False) if st.session_state.country_choice!='All countries' else slice(None)
    filtered = plant_df[mask] if st.session_state.country_choice!='All countries' else plant_df
    latin_list = sorted(filtered['latin_name'])
    choice = st.selectbox("Select a species by Latin name:", latin_list)
    if choice:
        info = filtered[filtered['latin_name']==choice].iloc[0].to_dict()
        display_plant_details(info)
