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
# PREPROCESS FUNCTION
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

if st.button("Generate Random"):
    st.session_state["random"] = True

# ================================
# INPUTS (TYPABLE)
# ================================
PSN = st.number_input("PSN", 0.0, 2000.0, 1500.0, step=10.0)
AVE_GROSS = st.number_input("Average Fluid Gross (bopd)", 0.0, 2000.0, 1500.0, step=10.0)
AVE_GAS = st.number_input("Average GAS (Mscfd)", 0.0, 100.0, 10.0, step=1.0)
PUMP_EFF = st.number_input("Pump Efficiency", 0.0, 1.0, 0.5, step=0.01)
OD_PUMP = st.number_input("Pump Outer Diameter (inch)", 1.0, 5.0, 2.5, step=0.1)
SL = st.number_input("Stroke Length (inch)", 0.0, 100.0, 50.0, step=1.0)
SPM = st.number_input("Stroke Per Minute", 0.0, 10.0, 5.0, step=0.1)
SM = st.number_input("Submergence (SM, in meter)", 0.0, 300.0, 100.0, step=1.0)
TORQUE = st.number_input("Torque", 0.0, 1.0, 0.5, step=0.01)
LOAD = st.number_input("Load", 0.0, 1.0, 0.5, step=0.01)
ROD_STRESS = st.number_input("Rod Stress", 0.0, 1.0, 0.5, step=0.01)
FREQ_OFF = st.number_input("Freq Off", 0.0, 50.0, 10.0, step=1.0)
HOUR_OFF = st.number_input("Hour Off", 0.0, 100.0, 20.0, step=1.0)

# KEEP 0/1 AS SELECTBOX
ROD_GUIDE = st.selectbox("Rod Guide", [0,1])
GASSY = st.selectbox("Gassy", [0,1])
PARAFFINIC = st.selectbox("Paraffinic", [0,1])
SCALE = st.selectbox("Scale", [0,1])

# CATEGORICAL
WELL_TYPE = st.selectbox("Well Type", ["VERTICAL","DEVIATED","UNKNOWN"])
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

    st.success(f"Predicted Lifetime: {prediction[0]:.2f} days")
