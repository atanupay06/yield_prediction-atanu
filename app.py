import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load the pre-trained pipeline and preprocessor ---
# The model and preprocessor files should be in the same directory as app.py
# Or a path to them should be provided.
pipe_rf = joblib.load('random_forest_pipeline.joblib')
preprocessor = joblib.load('preprocessor.joblib')

# --- 2. Extract unique values and ranges from original data (for UI population) ---
# These values are extracted once and hardcoded into the app.py for simplicity
# in deployment, avoiding the need to load df_clean in the Streamlit app itself.

# Categorical features options
state_names = ['Andaman and Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar']
district_names = ['NICOBARS', 'ANANTAPUR', 'PAPUM PARE', 'BHAGALPUR', 'DARBHANGA', 'KATIHAR', 'MADHEPURA', 'SHEOHAR', 'VAISHALI', 'KADAPA', 'SPSR NELLORE']
crop_years = ['1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
seasons = ['Autumn', 'Kharif', 'Rabi', 'Summer', 'Whole Year', 'Winter']
crops = ['Arecanut', 'Other Kharif pulses', 'Rice', 'Banana', 'Cashewnut', 'Coconut ', 'Dry ginger', 'Sugarcane', 'Sweet potato', 'Tapioca', 'Black pepper', 'Dry chillies', 'other oilseeds', 'Turmeric', 'Maize', 'Moong(Green Gram)', 'Urad', 'Arhar/Tur', 'Groundnut', 'Sunflower', 'Bajra', 'Castor seed', 'Cotton(lint)', 'Horse-gram', 'Jowar', 'Korra', 'Ragi', 'Tobacco', 'Gram', 'Wheat', 'Masoor', 'Sesamum', 'Linseed', 'Safflower', 'Onion', 'other misc. pulses', 'Samai', 'Small millets', 'Coriander', 'Potato', 'Other  Rabi pulses', 'Soyabean', 'Beans & Mutter(Vegetable)', 'Bhindi', 'Brinjal', 'Citrus Fruit', 'Cucumber', 'Grapes', 'Mango', 'Orange', 'other fibres', 'Other Fresh Fruits', 'Other Vegetables', 'Papaya', 'Pome Fruit', 'Tomato', 'Rapeseed &Mustard', 'Mesta', 'Cowpea(Lobia)', 'Lemon', 'Pome Granet', 'Sapota', 'Cabbage', 'Peas  (vegetable)', 'Niger seed', 'Bottle Gourd', 'Sannhamp', 'Varagu', 'Garlic', 'Ginger', 'Oilseeds total', 'Pulses total', 'Jute', 'Peas & beans (Pulses)', 'Blackgram', 'Paddy', 'Pineapple', 'Barley', 'Khesari', 'Guar seed']

# Numerical feature ranges and defaults (min, max, median from df_clean)
temp_min, temp_max, temp_default = 0, 49, 25
hum_min, hum_max, hum_default = 0, 99, 65
soil_min, soil_max, soil_default = 0, 99, 58
area_min, area_max, area_default = 0.2, 8580100.0, 100.0 # Using max from df_clean['Area']

# --- Streamlit App UI ---
st.title('Crop Yield Prediction App')
st.write('Enter the details below to predict crop yield and production.')

# Input widgets
st.header('Crop and Location Details')
state_name = st.selectbox('State Name', options=sorted(state_names))
district_name = st.selectbox('District Name', options=sorted(district_names))
crop_year = st.selectbox('Crop Year', options=sorted(crop_years, reverse=True))
season = st.selectbox('Season', options=sorted(seasons))
crop = st.selectbox('Crop', options=sorted(crops))

st.header('Environmental and Area Details')
temperature = st.slider('Temperature (°C)', min_value=float(temp_min), max_value=float(temp_max), value=float(temp_default), step=1.0)
humidity = st.slider('Humidity (%)', min_value=float(hum_min), max_value=float(hum_max), value=float(hum_default), step=1.0)
soil_moisture = st.slider('Soil Moisture (%)', min_value=float(soil_min), max_value=float(soil_max), value=float(soil_default), step=1.0)
area = st.slider('Area (Hectares)', min_value=float(area_min), max_value=float(area_max), value=float(area_default), step=10.0)

# Prediction button
if st.button('Predict Yield and Production'):
    # Create a DataFrame for prediction
    # Ensure column order matches X_train during model training
    input_data = pd.DataFrame([{
        'State_Name': state_name,
        'District_Name': district_name,
        'Crop_Year': crop_year, # Keep as string as per preprocessing
        'Season': season,
        'Crop': crop,
        'Temperature': temperature,
        'Humidity': humidity,
        'Soil_Moisture': soil_moisture,
        'Area': area
    }])

    # Predict Yield
    predicted_yield = pipe_rf.predict(input_data)[0]

    # Calculate Production
    predicted_production = predicted_yield * area

    st.success('Prediction Results:')
    st.write(f"**Predicted Yield:** {predicted_yield:.2f} units/hectare")
    st.write(f"**Predicted Production:** {predicted_production:.2f} units")
