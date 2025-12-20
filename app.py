import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================================
# 1. PAGE CONFIG & THEME (Applied to Whole App)
# =====================================================
st.set_page_config(page_title="Credit Risk AI", page_icon="💳", layout="wide")

themes = {
    "Midnight Blue 🌑": "linear-gradient(135deg,#000428,#004e92)",
    "Cyber Dark ⚡": "linear-gradient(135deg,#0f0f0f,#232526)",
    "Galaxy 🌌": "linear-gradient(135deg,#0f2027,#203a43,#2c5364)",
    "Forest 🌿": "linear-gradient(135deg,#134e5e,#71b280)",
    "Black Glass 🕶️": "#000000",
}

st.sidebar.title("🎨 UI Design")
theme_choice = st.sidebar.selectbox("Background Theme", list(themes.keys()))

st.markdown(f"""
<style>
    .stApp, [data-testid="stSidebar"], .stMain {{
        background: {themes[theme_choice]} !important;
        background-attachment: fixed;
        color: white;
    }}
    .metric-card {{
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 10px;
    }}
    h1, h2, h3, h4, p, label {{ color: white !important; }}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. DATA PIPELINE (Handles your 'data/' folder path)
# =====================================================
FILE_PATH = "data/Loan_Default.csv" if os.path.exists("data/Loan_Default.csv") else "Loan_Default.csv"

@st.cache_data(show_spinner=False)
def load_and_preprocess():
    if not os.path.exists(FILE_PATH): return None
    
    df = pd.read_csv(FILE_PATH)
    df = df.drop(columns=['ID', 'year'], errors='ignore')
    
    y = df['Status']
    X = df.drop(columns=['Status'])
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Imputation
    for col in num_cols: X[col] = X[col].fillna(X[col].median())
    for col in cat_cols: X[col] = X[col].fillna(X[col].mode()[0])
    
    # UI Helpers
    ui_options = {col: sorted(X[col].unique().tolist()) for col in cat_cols}
    num_stats = {col: {"min": float(X[col].min()), "max": float(X[col].max()), "med": float(X[col].median())} for col in num_cols}
    
    # Encoding & Scaling
    X_encoded = pd.get_dummies(X, columns=cat_cols)
    feature_names = X_encoded.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    return X_scaled, y, scaler, feature_names, ui_options, num_cols, cat_cols, num_stats

# SILENT INIT
data = load_and_preprocess()
if data is None:
    st.error("CSV file not found. Please check your 'data' folder.")
    st.stop()

X_s, y_s, scaler, feat_names, ui_opts, num_c, cat_c, num_s = data

# =====================================================
# 3. TRAINING BOTH MODELS (Regression restored)
# =====================================================
@st.cache_resource(show_spinner=False)
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Regression Model
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    
    # Random Forest Model
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    return lr, rf, X_test, y_test

lr_model, rf_model, X_test, y_test = train_models(X_s, y_s)

# =====================================================
# 4. UI MAIN
# =====================================================
st.title("💳 Credit Risk Assessment Intelligence")

# Global Model Selector (Restored)
st.sidebar.markdown("---")
model_choice = st.sidebar.radio("🤖 Select AI Engine", ["Logistic Regression", "Random Forest"])
active_model = lr_model if model_choice == "Logistic Regression" else rf_model

tab1, tab2, tab3 = st.tabs(["📊 Performance", "👤 Individual Assessment", "📂 Bulk Data"])

# --- TAB 1: PERFORMANCE ---
with tab1:
    y_pred = active_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><h4>Accuracy</h4><h2>{acc*100:.1f}%</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h4>Engine</h4><h2>{model_choice.split()[0]}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h4>Samples</h4><h2>{len(y_test):,}</h2></div>', unsafe_allow_html=True)
    
    st.write("#### Classification Report")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose(), use_container_width=True)

# --- TAB 2: INDIVIDUAL ASSESSMENT (Proper Labeling & Integer Menus) ---
with tab2:
    st.markdown("### 👤 Applicant Profiling")
    with st.form("assessment_form"):
        left, right = st.columns(2)
        u_input = {}

        with left:
            st.markdown("#### 💰 Financial Values")
            for col in num_c:
                label = col.replace('_', ' ').title()
                # Specific Slider for Credit Score
                if "Score" in label:
                    u_input[col] = st.slider(label, 300, 900, int(num_s[col]['med']))
                else:
                    u_input[col] = st.number_input(label, value=num_s[col]['med'])

        with right:
            st.markdown("#### 📋 Qualitative Factors")
            for col in cat_c:
                u_input[col] = st.selectbox(f"{col.replace('_', ' ').title()}", options=ui_opts[col])

        st.write("###")
        if st.form_submit_button("🚀 RUN PREDICTION", use_container_width=True):
            # Process & Predict
            in_df = pd.DataFrame([u_input])
            in_enc = pd.get_dummies(in_df).reindex(columns=feat_names, fill_value=0)
            in_scl = scaler.transform(in_enc)
            
            res = active_model.predict(in_scl)[0]
            prob = active_model.predict_proba(in_scl)[0][1]
            
            st.markdown("---")
            rl, rr = st.columns(2)
            if res == 1: rl.error("### 🚩 HIGH RISK")
            else: rl.success("### ✅ LOW RISK")
            rr.metric("Probability of Default", f"{prob*100:.1f}%")

# --- TAB 3: BULK ---
with tab3:
    st.markdown("### 📂 Bulk Processing")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        st.success("File uploaded. Ready for processing.")