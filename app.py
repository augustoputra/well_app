import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# LOAD FILES
# ================================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
num_imputer = joblib.load("num_imputer.pkl")
cat_imputer = joblib.load("cat_imputer.pkl")

num_cols = ['PSN','AVE_GROSS','AVE_GAS','PUMP_EFF','OD_PUMP','SL','SPM','SM',
            'TORQUE','LOAD','ROD_STRESS','FREQ_OFF','HOUR_OFF','ROD_GUIDE',
            'GASSY','PARAFFINIC','SCALE']

cat_cols = ['WELL_TYPE','DHS']

# ================================
# PREPROCESS FUNCTION (FROM YOUR NOTEBOOK)
# ================================
def preprocess_data(data, num_cols, cat_cols, num_imputer, cat_imputer, cat_encoder, scaler):
    X_num = data[num_cols]
    X_cat = data[cat_cols]

    X_num = num_imputer.transform(X_num)
    X_num = pd.DataFrame(X_num, columns=num_cols)

    X_cat = cat_imputer.transform(X_cat)
    X_cat = pd.DataFrame(X_cat, columns=cat_cols)

    X_cat_encoded = cat_encoder.transform(X_cat)
    X_cat_encoded = pd.DataFrame(
        X_cat_encoded,
        columns=cat_encoder.get_feature_names_out(cat_cols)
    )

    X_concat = pd.concat([X_num, X_cat_encoded], axis=1)

    X_scaled = scaler.transform(X_concat)

    return X_scaled

# ================================
# UI
# ================================
st.title("Well Lifetime Prediction")

# RANDOM BUTTON
if st.button("Generate Random"):
    st.session_state["random"] = True

# INPUTS
PSN = st.slider("PSN", 0.0, 500.0, 1500.0)
AVE_GROSS = st.slider("AVE_GROSS", 0.0, 500.0, 1500.0)
AVE_GAS = st.slider("AVE_GAS", 0.0, 100.0, 10.0)
PUMP_EFF = st.slider("PUMP_EFF", 0.0, 1.0, 0.5)
OD_PUMP = st.slider("OD_PUMP", 1.0, 5.0, 2.5)
SL = st.slider("SL", 0.0, 100.0, 50.0)
SPM = st.slider("SPM", 0.0, 10.0, 5.0)
SM = st.slider("SM", 0.0, 300.0, 100.0)
TORQUE = st.slider("TORQUE", 0.0, 1.0, 0.5)
LOAD = st.slider("LOAD", 0.0, 1.0, 0.5)
ROD_STRESS = st.slider("ROD_STRESS", 0.0, 1.0, 0.5)
FREQ_OFF = st.slider("FREQ_OFF", 0.0, 50.0, 10.0)
HOUR_OFF = st.slider("HOUR_OFF", 0.0, 100.0, 20.0)

ROD_GUIDE = st.selectbox("ROD_GUIDE", [0,1])
GASSY = st.selectbox("GASSY", [0,1])
PARAFFINIC = st.selectbox("PARAFFINIC", [0,1])
SCALE = st.selectbox("SCALE", [0,1])

WELL_TYPE = st.selectbox("WELL_TYPE", ["VERTICAL","DEVIATED","UNKNOWN"])
DHS = st.selectbox("DHS", ["GACT","SANDTRAP","SANDTRAP_SHROUD","HYBRID", 'SCREEN', 'UNKNOWN'])

# ================================
# PREDICT
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

    processed = preprocess_data(input_df, num_cols, cat_cols,
                                num_imputer, cat_imputer, encoder, scaler)

    prediction = model.predict(processed)

    st.success(f"Predicted Lifetime: {prediction[0]:.2f}")