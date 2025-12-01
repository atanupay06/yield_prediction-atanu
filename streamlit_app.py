# ----------------------------------------------------------
# 🌾 Crop Yield & Production Prediction App (Ultimate UI)
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------- CONFIG -----------------------
MODEL_PATH = Path("decision_tree_pipeline.joblib")  # your trained pipeline
# -----------------------------------------------------

# ---------------------- UI STYLING -------------------
st.set_page_config(page_title="🌾 Crop Yield Predictor", layout="wide")

st.markdown("""
<style>
/* Smooth background */
.stApp {
    background: #0e1929;
}

/* Title */
.big-title {
    font-size: 40px !important;
    font-weight: 800 !important;
    padding: 10px;
    color: #2F855A;
    text-align: center;
}

/* Card Styling */
.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 6px 25px rgba(0,0,0,0.10);
    margin-bottom: 20px;
}

/* Prediction result style */
.result-card {
    background: #e6fffa;
    padding: 25px;
    border-left: 6px solid #2c7a7b;
    border-radius: 10px;
    font-size: 22px;
    color: #234e52;
    font-weight: 600;
}

/* Button style */
.stButton>button {
    background: linear-gradient(90deg, #319795, #2c7a7b);
    color: white;
    padding: 12px 22px;
    border-radius: 8px;
    font-size: 18px;
    transition: 0.2s;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #2c7a7b, #285e61);
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🌾 AI-Powered Crop Yield & Production Predictor</div>", unsafe_allow_html=True)

# ------------------------------------------------------
# DATA FOR DROPDOWNS (Cascading State → District)
# ------------------------------------------------------

STATE_DISTRICTS = {
    "Andaman and Nicobar Islands": [
        "NICOBARS", "NORTH AND MIDDLE ANDAMAN", "SOUTH ANDAMANS"
    ],
    "Andhra Pradesh": [
        "ANANTAPUR","CHITTOOR","EAST GODAVARI","GUNTUR","KADAPA",
        "KRISHNA","KURNOOL","PRAKASAM","SPSR NELLORE","SRIKAKULAM",
        "VISAKHAPATANAM","VIZIANAGARAM","WEST GODAVARI"
    ],
    "Arunachal Pradesh": [
        "ANJAW","CHANGLANG","DIBANG VALLEY","EAST KAMENG","EAST SIANG",
        "KURUNG KUMEY","LOHIT","LONGDING","LOWER DIBANG VALLEY",
        "LOWER SUBANSIRI","NAMSAI","PAPUM PARE","TAWANG","TIRAP",
        "UPPER SIANG","UPPER SUBANSIRI","WEST KAMENG","WEST SIANG"
    ],
    "Assam": [
        "BAKSA","BARPETA","BONGAIGAON","CACHAR","CHIRANG","DARRANG",
        "DHEMAJI","DHUBRI","DIBRUGARH","DIMA HASAO","GOALPARA",
        "GOLAGHAT","HAILAKANDI","JORHAT","KAMRUP","KAMRUP METRO",
        "KARBI ANGLONG","KARIMGANJ","KOKRAJHAR","LAKHIMPUR",
        "MARIGAON","NAGAON","NALBARI","SIVASAGAR","SONITPUR",
        "TINSUKIA","UDALGURI"
    ],
    "Bihar": [
        "ARARIA","ARWAL","AURANGABAD","BANKA","BEGUSARAI","BHAGALPUR",
        "BHOJPUR","BUXAR","DARBHANGA","GAYA","GOPALGANJ","JAMUI",
        "JEHANABAD","KAIMUR (BHABUA)","KATIHAR","KHAGARIA","KISHANGANJ",
        "LAKHISARAI","MADHEPURA","MADHUBANI","MUNGER","MUZAFFARPUR",
        "NALANDA","NAWADA","PASHCHIM CHAMPARAN","PATNA","PURBI CHAMPARAN",
        "PURNIA","ROHTAS","SAHARSA","SAMASTIPUR","SARAN","SHEIKHPURA",
        "SHEOHAR","SITAMARHI","SIWAN","SUPAUL","VAISHALI"
    ],
    "Chandigarh": ["CHANDIGARH"],
    "Chhattisgarh": [
        "BALOD","BALODA BAZAR","BALRAMPUR","BASTAR","BEMETARA","BIJAPUR",
        "BILASPUR","DANTEWADA","DHAMTARI","DURG","GARIYABAND","JANJGIR-CHAMPA"
    ]
}

SEASONS = ["Kharif", "Whole Year", "Autumn", "Rabi", "Summer", "Winter"]

CROPS = [
    'Arecanut','Other Kharif pulses','Rice','Banana','Cashewnut','Coconut',
    'Dry ginger','Sugarcane','Sweet potato','Tapioca','Black pepper',
    'Dry chillies','other oilseeds','Turmeric','Maize','Moong(Green Gram)',
    'Urad','Arhar/Tur','Groundnut','Sunflower','Bajra','Castor seed',
    'Cotton(lint)','Horse-gram','Jowar','Korra','Ragi','Tobacco','Gram',
    'Wheat','Masoor','Sesamum','Linseed','Safflower','Onion',
    'other misc. pulses','Samai','Small millets','Coriander','Potato',
    'Other  Rabi pulses','Soyabean','Beans & Mutter(Vegetable)','Bhindi',
    'Brinjal','Citrus Fruit','Cucumber','Grapes','Mango','Orange',
    'other fibres','Other Fresh Fruits','Other Vegetables','Papaya',
    'Pome Fruit','Tomato','Rapeseed &Mustard','Mesta','Cowpea(Lobia)',
    'Lemon','Pome Granet','Sapota','Cabbage','Peas  (vegetable)',
    'Niger seed','Bottle Gourd','Sannhamp','Varagu','Garlic','Ginger',
    'Oilseeds total','Pulses total','Jute','Peas & beans (Pulses)',
    'Blackgram','Paddy','Pineapple','Barley','Khesari','Guar seed'
]

# -----------------------------------------------------
# LOAD MODEL WITH ERROR HANDLING
# -----------------------------------------------------
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error("⚠️ Model found but failed to load! Check version mismatch.")
            st.caption(str(e))
            return None
    else:
        st.warning("⚠️ No model found, app will run in DEMO MODE!")
        return None

model = load_model()

# -----------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------
left, right = st.columns([1, 1])

# ----------------------- LEFT PANEL -----------------------
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📍 Enter Location & Crop Details")

    state = st.selectbox("Select State", list(STATE_DISTRICTS.keys()))
    district = st.selectbox("Select District", STATE_DISTRICTS[state])
    season = st.selectbox("Select Season", SEASONS)
    crop = st.selectbox("Select Crop", CROPS)
    year = st.number_input("Crop Year", min_value=1990, max_value=2050, value=2020)

    st.subheader("🌡️ Environment Inputs")
    temp = st.slider("Temperature (°C)", 10, 50, 30)
    hum = st.slider("Humidity (%)", 10, 100, 50)
    soil = st.slider("Soil Moisture", 0, 100, 45)
    area = st.number_input("Enter Area (Units)", min_value=0.1, value=100.0)

    predict_btn = st.button("🚀 Predict Yield & Production")
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------- RIGHT PANEL -----------------------
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Output")

    if predict_btn:
        row = {
            "State_Name": state,
            "District_Name": district,
            "Crop_Year": str(year),
            "Season": season,
            "Crop": crop,
            "Temperature": temp,
            "Humidity": hum,
            "Soil_Moisture": soil,
            "Area": area
        }

        df_input = pd.DataFrame([row])

        if model:
            try:
                pred_yield = float(model.predict(df_input)[0])
                pred_production = pred_yield * area

                st.markdown(f"""
                    <div class="result-card">
                        🌱 **Predicted Yield:** {pred_yield:.4f} per unit  
                        📦 **Total Production:** {pred_production:.2f} units  
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error("❌ Prediction failed — check your model pipeline!")
                st.caption(str(e))

        else:
            # DEMO MODE (if model not loaded)
            demo_yield = (temp * 0.01) + (soil * 0.005)
            demo_prod = demo_yield * area
            st.info("⚠️ Demo Mode: Model not loaded, showing simulated result.")
            st.markdown(f"""
                <div class="result-card">
                    🌱 **Estimated Yield:** {demo_yield:.4f}  
                    📦 **Estimated Production:** {demo_prod:.2f}
                </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit by Atanu Paul-2025")

