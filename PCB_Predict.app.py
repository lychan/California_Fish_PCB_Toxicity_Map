# --- Main Streamlit App ---

# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

### LAYOUT CONFIG
st.set_page_config(
    page_title="California Fish PCB Toxicity Prediction Map 🐟", 
    page_icon="🐟", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# Title and Text with Fish Emoji 
st.title("California Fish PCB Toxicity Prediction Map 🐟")
st.write("This map displays the predicted locations where fish exceed the toxicity threshold of 75 ng/g (ww) for polychlorinated biphenyls (PCBs).")

### DATA LOADING
# A. Define functions to load data
@st.cache_data
def load_data():
    PCB_data = pd.read_csv('PCB_model_data.csv')
    PCB_data = PCB_data.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})    
    return PCB_data

@st.cache_data
def load_station_coordinates():
    station_data = pd.read_csv('Station_ID_Lat_Lon.csv')
    station_data = station_data.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}) 
    return station_data

# B. Load Data
PCB_data = load_data()
station_data = load_station_coordinates()

### MODEL INFERENCE

# A. Load the model using joblib
@st.cache_resource
def load_model():
    return joblib.load('PCB_rf_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# B. Set up input field
# List species names from the columns (only columns starting with 'CommonName_')
species_columns = [col for col in PCB_data.columns if col.startswith('CommonName_')]  # Filter columns by prefix

# --- User Dropdown for Species Selection ---
selected_species = st.selectbox("Select a Fish Species:", species_columns)  # Display species names with prefix

@st.cache_data
def filter_data(species_name):
    return PCB_data[PCB_data[species_name] == 1].copy()  # Filter based on the full species column name

# Filter data based on selected species (assuming species columns have binary values: 1 for presence)
filtered_data = filter_data(selected_species)

# C. Prepare data for prediction

feature_columns = [col for col in PCB_data.columns if col not in ['PCB_threshold']]
X = filtered_data[feature_columns]

# D. Merge predictions with latitude and longitude for map visualization

# Merge filtered data with station coordinates for visualization, based on 'Station ID'
station_data = station_data.drop_duplicates(subset=['Station ID'])

# Merge filtered data with station coordinates for visualization
merged_data = pd.merge(filtered_data, station_data, on='Station ID', how='left')

# --- Ensure X and merged_data have the same number of rows ---
# Check if the length of X matches the length of merged_data
if len(X) != len(merged_data):
    st.error(f"Mismatch in rows between the feature data (X) and merged data: {len(X)} vs {len(merged_data)}")
    st.stop()

# --- Make Predictions on Merged Data ---
with st.spinner("Running predictions..."):
    try:
        # Make predictions with the correct feature set
        predictions = model.predict(X)
        merged_data['prediction'] = predictions  # Add predictions to the merged data
        st.success("Predictions completed successfully!")
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.stop()


# --- Map Visualization ---

visual_chart = merged_data[['Station ID', 'lat', 'lon', 'prediction']]
st.write(visual_chart)  # Display the table with Station ID, lat, lon, and predictions

# Filter to show only positive predictions (prediction == 1)
positive_data = merged_data[merged_data['prediction'] == 1]

# Drop rows with NaN values in 'lat' or 'lon' before proceeding
positive_data = positive_data.dropna(subset=['lat', 'lon'])

# Sample data to limit points on the map (up to 1000 points)
sampled_data = positive_data.sample(n=min(1000, len(positive_data)), random_state=42)

# --- Map Creation ---
# If there are any positive predictions, create a map centered around the mean latitude and longitude of the positive data
if len(sampled_data) > 0:
    m = folium.Map(location=[sampled_data['lat'].mean(), sampled_data['lon'].mean()], zoom_start=5)
else:
    # If no positive predictions, create a map centered on the mean of the full data
    m = folium.Map(location=[merged_data['lat'].mean(), merged_data['lon'].mean()], zoom_start=5)

# Add MarkerCluster to group close points together (only if positive data exists)
if len(sampled_data) > 0:
    marker_cluster = MarkerCluster().add_to(m)
    
    # Loop through the data and add markers for each species (using full species column name with prefix)
    for _, row in sampled_data.iterrows():
        species_column = selected_species  # Use the full species column name with prefix
        if row[species_column] == 1:  # Check if the species is present in the row
            marker_color = 'green'  # You can set a default color or use logic for specific species

            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                color=marker_color,  # Use species color
                fill=True,
                fill_opacity=0.6,
                popup=f"Station ID: {row['Station ID']}, Species: {selected_species}, Toxicity: Positive"  # Show species and toxicity
            ).add_to(marker_cluster)

# Display the map in Streamlit
folium_static(m)

# If no positive predictions were found, show a message
if len(sampled_data) == 0:
    st.warning("No positive predictions available for map display.")
