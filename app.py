import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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


@st.cache_data
def load_raw_data():
    df = pd.read_csv("data.csv")
    df = df.drop_duplicates()
    return df


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
# HELPER: Yes/No → 0/1
# ================================
def yn(val):
    return 1 if val == "Yes" else 0

# ================================
# UI
# ================================
st.title("Well Lifetime Prediction")

st.markdown("""
<style>
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
}
section.main * {
    font-family: sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ================================
# TABS
# ================================
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Feature Distribution"])

# ================================
# TAB 1: PREDICTION
# ================================
with tab1:

    st.markdown("#### Well & Pump Parameters")
    col_a, col_b = st.columns(2)

    with col_a:
        PSN        = st.slider("PSN",                          0.0,   2500.0, 1500.0,  step=10.0)
        AVE_GROSS  = st.slider("Average Fluid Gross (bopd)",   0.0,   2500.0, 1500.0,  step=10.0)
        AVE_GAS    = st.slider("Average GAS (Mscfd)",          0.0,   1000.0,   10.0,  step=1.0)
        PUMP_EFF   = st.slider("Pump Efficiency",              0.0,      1.0,    0.5,  step=0.01)
        OD_PUMP    = st.slider("Pump Outer Diameter (inch)",   1.0,      5.0,    2.5,  step=0.1)
        SL         = st.slider("Stroke Length (inch)",         0.0,    200.0,   50.0,  step=1.0)
        SPM        = st.slider("Stroke Per Minute",            0.0,     13.0,    5.0,  step=0.1)
        SM         = st.slider("Submergence (meter)",          0.0,   3000.0,  100.0,  step=10.0)

    with col_b:
        TORQUE     = st.slider("Torque",                       0.0,      1.0,    0.5,  step=0.01)
        LOAD       = st.slider("Load",                         0.0,      1.0,    0.5,  step=0.01)
        ROD_STRESS = st.slider("Rod Stress",                   0.0,      1.0,    0.5,  step=0.01)
        FREQ_OFF   = st.slider("Freq Off",                     0.0,    500.0,   10.0,  step=1.0)
        HOUR_OFF   = st.slider("Hour Off",                     0.0,   9000.0,   20.0,  step=10.0)

    st.markdown("#### Well Conditions")
    col_c, col_d, col_e, col_f = st.columns(4)
    ROD_GUIDE_yn  = col_c.selectbox("Rod Guide",   ["No", "Yes"])
    GASSY_yn      = col_d.selectbox("Gassy",       ["No", "Yes"])
    PARAFFINIC_yn = col_e.selectbox("Paraffinic",  ["No", "Yes"])
    SCALE_yn      = col_f.selectbox("Scale",       ["No", "Yes"])

    ROD_GUIDE  = yn(ROD_GUIDE_yn)
    GASSY      = yn(GASSY_yn)
    PARAFFINIC = yn(PARAFFINIC_yn)
    SCALE      = yn(SCALE_yn)

    st.markdown("#### Well Classification")
    col_g, col_h = st.columns(2)
    WELL_TYPE = col_g.selectbox("Well Type", ["VERTICAL", "DEVIATED", "UNKNOWN"])
    DHS       = col_h.selectbox("DHS", ["GACT", "SANDTRAP", "SANDTRAP_SHROUD", "HYBRID", "SCREEN", "UNKNOWN"])

    st.markdown("---")
    if st.button("🔮 Predict Lifetime", use_container_width=True):

        input_df = pd.DataFrame([{
            'PSN': PSN, 'AVE_GROSS': AVE_GROSS, 'AVE_GAS': AVE_GAS,
            'PUMP_EFF': PUMP_EFF, 'OD_PUMP': OD_PUMP, 'SL': SL,
            'SPM': SPM, 'SM': SM, 'TORQUE': TORQUE, 'LOAD': LOAD,
            'ROD_STRESS': ROD_STRESS, 'FREQ_OFF': FREQ_OFF, 'HOUR_OFF': HOUR_OFF,
            'ROD_GUIDE': ROD_GUIDE, 'GASSY': GASSY, 'PARAFFINIC': PARAFFINIC,
            'SCALE': SCALE, 'WELL_TYPE': WELL_TYPE, 'DHS': DHS
        }])

        processed = preprocess_data(input_df)
        prediction = model.predict(processed)[0]

        # Save to session state for Tab 2
        st.session_state['user_input'] = {
            'PSN': PSN, 'AVE_GROSS': AVE_GROSS, 'AVE_GAS': AVE_GAS,
            'PUMP_EFF': PUMP_EFF, 'OD_PUMP': OD_PUMP, 'SL': SL,
            'SPM': SPM, 'SM': SM, 'TORQUE': TORQUE, 'LOAD': LOAD,
            'ROD_STRESS': ROD_STRESS, 'FREQ_OFF': FREQ_OFF, 'HOUR_OFF': HOUR_OFF,
            'ROD_GUIDE': ROD_GUIDE, 'GASSY': GASSY, 'PARAFFINIC': PARAFFINIC,
            'SCALE': SCALE
        }

        # BIG KPI DISPLAY
        st.markdown(f"""
        <div style="
            background-color: #0f5132;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 20px;
        ">
            <div style="font-size: 20px; color: #a7f3d0; font-weight: 600;">
                Predicted Lifetime
            </div>
            <div style="font-size: 56px; color: #00ff88; font-weight: 900;">
                {prediction:.0f} Days
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("💡 Go to the **Feature Distribution** tab to see where your inputs sit in the data.")

        # MODEL PERFORMANCE
        if HAS_TRAIN:
            st.subheader("Model Performance")

            r2   = r2_score(y_train, y_train_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

            model_type = type(model).__name__
            if "XGB" in model_type:       model_type = "XGBoost"
            elif "LGBM" in model_type:    model_type = "LightGBM"
            elif "RandomForest" in model_type: model_type = "Random Forest"

            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score",   f"{r2:.3f}")
            col2.metric("RMSE",       f"{rmse:.2f}")
            col3.metric("Model Type", model_type)

            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.scatter(y_train, y_train_pred, alpha=0.6)
            min_val = min(np.min(y_train), np.min(y_train_pred))
            max_val = max(np.max(y_train), np.max(y_train_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r-')
            ax.set_xlim(0, 800); ax.set_ylim(0, 800)
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
            plt.close(fig)

# ================================
# TAB 2: FEATURE DISTRIBUTION
# ================================
with tab2:
    st.subheader("Feature Distributions")

    user_input = st.session_state.get('user_input', None)

    if user_input is None:
        st.info("👈 Go to the **Prediction** tab, fill in your inputs and click **Predict** — your values will then be highlighted in red on the charts below.")

    try:
        df_raw     = load_raw_data()
        X_train_num = df_raw[num_cols]

        cols = st.columns(2)
        for i, col in enumerate(X_train_num.columns):
            fig, ax = plt.subplots(figsize=(6, 3))

            sns.histplot(X_train_num[col].dropna(), ax=ax, color='steelblue')
            ax.set_title(f'Distribution of {col}')
            ax.set_xlim(left=0)

            if user_input is not None and col in user_input:
                val     = user_input[col]
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]

                ax.axvline(x=val, color='red', linewidth=2,
                           linestyle='--', label=f'Your input: {val}')
                ax.axvspan(val - x_range * 0.02,
                           val + x_range * 0.02,
                           alpha=0.25, color='red')
                ax.legend(fontsize=8)

            plt.tight_layout()
            cols[i % 2].pyplot(fig)
            plt.close(fig)

    except Exception as e:
        st.warning(f"Could not load data for charts: {e}")
