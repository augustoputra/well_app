import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ================================
# CACHE EVERYTHING (IMPORTANT 🚀)
# ================================
@st.cache_resource
def load_all():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    num_imputer = joblib.load("num_imputer.pkl")
    cat_imputer = joblib.load("cat_imputer.pkl")

    try:
        y_train = joblib.load("y_train.pkl")
        y_train_pred = joblib.load("y_train_pred.pkl")
        HAS_TRAIN = True
    except:
        y_train, y_train_pred = None, None
        HAS_TRAIN = False

    return model, scaler, encoder, num_imputer, cat_imputer, y_train, y_train_pred, HAS_TRAIN


model, scaler, encoder, num_imputer, cat_imputer, y_train, y_train_pred, HAS_TRAIN = load_all()

# ================================
# CONFIG
# ================================
num_cols = ['PSN','AVE_GROSS','AVE_GAS','PUMP_EFF','OD_PUMP','SL','SPM','SM',
            'TORQUE','LOAD','ROD_STRESS','FREQ_OFF','HOUR_OFF','ROD_GUIDE',
            'GASSY','PARAFFINIC','SCALE']

cat_cols = ['WELL_TYPE','DHS']

# ================================
# PREPROCESS FUNCTION
# ================================
def preprocess_data(data):
    X_num = data[num_cols]
    X_cat = data[cat_cols]

    X_num = num_imputer.transform(X_num)
    X_num = pd.DataFrame(X_num, columns=num_cols)

    X_cat = cat_imputer.transform(X_cat)
    X_cat = pd.DataFrame(X_cat, columns=cat_cols)

    X_cat_encoded = encoder.transform(X_cat)
    X_cat_encoded = pd.DataFrame(
        X_cat_encoded,
        columns=encoder.get_feature_names_out(cat_cols)
    )

    X_concat = pd.concat([X_num, X_cat_encoded], axis=1)
    X_scaled = scaler.transform(X_concat)

    return X_scaled

# ================================
# UI
# ================================
st.title("Well Lifetime Prediction")

# ================================
# INPUTS
# ================================
# ================================
# INPUTS (DYNAMIC RANGES)
# ================================
PSN = st.slider("PSN", 0.0, 2500.0, 1500.0)
AVE_GROSS = st.slider("Average Fluid Gross (bopd)", 0.0, 2500.0, 1500.0)
AVE_GAS = st.slider("Average GAS (Mscfd)", 0.0, 1000.0, 10.0)
PUMP_EFF = st.slider("Pump Efficiency", 0.0, 1.0, 0.5)
OD_PUMP = st.slider("Pump Outer Diameter (inch)", 1.0, 5.0, 2.5)
SL = st.slider("Stroke Length (inch)", 0.0, 200.0, 50.0)
SPM = st.slider("Stroke Per Minute", 0.0, 13.0, 5.0)
SM = st.slider("Submergence (SM, in meter)", 0.0, 3000.0, 100.0)
TORQUE = st.slider("Torque", 0.0, 1.0, 0.5)    
LOAD = st.slider("Load", 0.0, 1.0, 0.5)
ROD_STRESS = st.slider("Rod Stress", 0.0, 1.0, 0.5)
FREQ_OFF = st.slider("Freq Off", 0.0, 500.0, 10.0)
HOUR_OFF = st.slider("Hour Off", 0.0, 9000.0, 20.0)

ROD_GUIDE = st.selectbox("Rod Guide", [0,1])
GASSY = st.selectbox("Gassy", [0,1])
PARAFFINIC = st.selectbox("Paraffinic", [0,1])
SCALE = st.selectbox("Scale", [0,1])

WELL_TYPE = st.selectbox("Well Type", ["VERTICAL","DEVIATED","UNKNOWN"])
DHS = st.selectbox("DHS", ["GACT","SANDTRAP","SANDTRAP_SHROUD","HYBRID","SCREEN","UNKNOWN"])

# ================================
# PREDICT (FAST)
# ================================
if st.button("Predict"):

    input_df = pd.DataFrame([{
        'PSN': PSN,
        'AVE_GROSS': AVE_GROSS,
        'AVE_GAS': AVE_GAS,
        'PUMP_EFF': PUMP_EFF,
        'OD_PUMP': OD_PUMP,
        'SL': SL,
        'SPM': SPM,
        'SM': SM,
        'TORQUE': TORQUE,
        'LOAD': LOAD,
        'ROD_STRESS': ROD_STRESS,
        'FREQ_OFF': FREQ_OFF,
        'HOUR_OFF': HOUR_OFF,
        'ROD_GUIDE': ROD_GUIDE,
        'GASSY': GASSY,
        'PARAFFINIC': PARAFFINIC,
        'SCALE': SCALE,
        'WELL_TYPE': WELL_TYPE,
        'DHS': DHS
    }])

    processed = preprocess_data(input_df)
    prediction = model.predict(processed)[0]

    # KPI
    st.metric("Predicted Lifetime (days)", f"{prediction:.0f}")

# ================================
# ACTUAL vs PREDICTED (PRO CHART)
# ================================
if HAS_TRAIN:
    st.subheader("Actual vs Predicted (Training Data)")

    fig, ax = plt.subplots()

    # Scatter
    ax.scatter(y_train, y_train_pred, alpha=0.6)

    # Diagonal line (perfect prediction)
    min_val = min(min(y_train), min(y_train_pred))
    max_val = max(max(y_train), max(y_train_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r-')

    ax.set_xlim(0, 800)
    ax.set_ylim(0, 800)

    # Labels
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")

    st.pyplot(fig)
