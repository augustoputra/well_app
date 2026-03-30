import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

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

    X_cat_enco
