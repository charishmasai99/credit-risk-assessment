import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# 1. PAGE CONFIG & STYLING
# =====================================================
st.set_page_config(page_title="Credit Risk AI", page_icon="💳", layout="wide")
# =====================================================
# 1. PAGE CONFIG & STYLING (UPDATED)
# =====================================================
st.set_page_config(page_title="Credit Risk AI", page_icon="💳", layout="wide")

st.markdown("""
<style>
    /* 1. Base styling */
    .stApp, [data-testid="stSidebar"], .stMain {
        background-attachment: fixed !important;
    }
    
    /* 2. Global text color */
    h1, h2, h3, h4, p, label, li, .stSelectbox p, .stNumberInput p, 
    [data-testid="stMarkdownContainer"] p, 
    .stAlert p, .stAlert div { 
        color: white !important; 
    }

    /* 3. FIX: Applicant Profiling Header Visibility */
    /* Targets the specific H3 inside the main container */
    [data-testid="stMain"] h3 {
        color: #FFFFFF !important;
        opacity: 1 !important;
        display: block !important;
    }

    /* 4. FIX: Instruction Box Visibility */
    /* We create a specific class for the instruction div */
    .custom-instruction-box {
        background-color: rgba(0, 123, 255, 0.2) !important;
        border-left: 4px solid #007BFF !important;
        padding: 12px !important;
        margin: 10px 0 !important;
        border-radius: 4px !important;
    }

    /* Ensures text inside the instruction box is bright white */
    .custom-instruction-box, .custom-instruction-box strong, .custom-instruction-box span {
        color: #FFFFFF !important;
    }
    /* FIX: CLEAR FORM button text visibility */
    div.stButton > button {
        background-color: #444 !important; /* Darker background so white text pops */
        color: white !important;
        font-weight: 700 !important;
        width: 100% !important;
    }

    /* Required field styling */
    .required-asterisk {
        color: #FF4B4B !important;
        font-weight: bold !important;
        font-size: 1.1em !important;
    }
    
    /* Rest of your existing styles remain exactly as they were */
    .validation-error {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 4px solid #FF4B4B;
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
    }
    
    div.stFormSubmitButton > button {
        background-color: #2563EB !important;
        color: white !important;
        font-weight: 700 !important;
        height: 3.5em !important;
        width: 100% !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. CONFIGURATION & DATA SETUP
# =====================================================

REQUIRED_FIELDS = {
    "loan_amount": {"min_value": 1, "label": "Loan Amount"},
    "income": {"min_value": 1, "label": "Monthly Income"},
    "property_value": {"min_value": 1, "label": "Property Value"},
    "Credit_Score": {"min_value": 300, "label": "Credit Score"},
    "Gender": {"label": "Gender"},
    "loan_type": {"label": "Loan Type"},
    "loan_purpose": {"label": "Loan Purpose"}
}

FIELD_EXPLANATIONS = {
    "approv_in_adv": {
        "title": "Pre-approval Status",
        "options": {
            "Yes": "✅ **Pre-approved**: Application was pre-screened and conditionally approved",
            "No": "❌ **Not pre-approved**: Standard application process without pre-screening"
        }
    },
    "loan_type": {
        "title": "Loan Product Type",
        "options": {
            "type1": "🏠 **Personal Loan**: Unsecured loan for personal expenses",
            "type2": "🏡 **Home Loan**: Secured loan for property purchase",
            "type3": "🎓 **Education Loan**: Specialized loan for educational expenses"
        }
    },
    "loan_purpose": {
        "title": "Purpose of Loan",
        "options": {
            "p1": "💼 **Business**: Starting or expanding a business venture",
            "p2": "📚 **Education**: Funding education or skill development",
            "p3": "🏥 **Medical**: Healthcare expenses or medical emergencies"
        }
    },
    "Credit_Worthiness": {
        "title": "Credit History Assessment",
        "options": {
            "l1": "🔴 **Low**: Poor credit history, high risk",
            "l2": "🟡 **Medium**: Average credit history, moderate risk",
            "l3": "🟢 **High**: Excellent credit history, low risk"
        }
    }
}

FILE_PATH = "data/Loan_Default.csv" if os.path.exists("data/Loan_Default.csv") else "Loan_Default.csv"

@st.cache_data(show_spinner=False)
def load_and_preprocess():
    if not os.path.exists(FILE_PATH): return None
    df = pd.read_csv(FILE_PATH)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df = df.drop(columns=['ID', 'year'], errors='ignore')
    y = df['Status']
    X = df.drop(columns=['Status'])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in num_cols: X[col] = X[col].fillna(X[col].median())
    for col in cat_cols: X[col] = X[col].fillna(X[col].mode()[0])
    ui_options = {col: sorted(X[col].unique().tolist()) for col in cat_cols}
    num_stats = {col: {"med": float(X[col].median())} for col in num_cols}
    X_encoded = pd.get_dummies(X, columns=cat_cols)
    feature_names = X_encoded.columns.tolist()
    scaler = StandardScaler().fit(X_encoded)
    X_scaled = scaler.transform(X_encoded)
    return X_scaled, y, scaler, feature_names, ui_options, num_cols, cat_cols, num_stats

data = load_and_preprocess()
if data is None:
    st.error("CSV file not found. Please ensure Loan_Default.csv is in the directory.")
    st.stop()

X_s, y_s, scaler, feat_names, ui_opts, num_c, cat_c, num_s = data

@st.cache_resource(show_spinner=False)
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train)
    return lr, rf, X_test, y_test

lr_model, rf_model, X_test, y_test = train_models(X_s, y_s)

# =====================================================
# 3. UTILITY & VALIDATION FUNCTIONS
# =====================================================

def validate_form_data(form_data):
    errors = []
    for field, rules in REQUIRED_FIELDS.items():
        if field in form_data:
            value = form_data[field]
            if field in num_c:
                if value is None or value <= 0:
                    errors.append(f"❌ {rules['label']} is required and must be greater than 0")
                elif "min_value" in rules and value < rules["min_value"]:
                    errors.append(f"❌ {rules['label']} must be at least {rules['min_value']}")
            elif field in cat_c:
                if value is None or value == "":
                    errors.append(f"❌ {rules['label']} selection is required")
    return len(errors) == 0, errors

# =====================================================
# 4. UI SETUP & SIDEBAR
# =====================================================

themes = {
    "Midnight Blue 🌑": "linear-gradient(135deg,#000428,#004e92)",
    "Cyber Dark ⚡": "linear-gradient(135deg,#0f0f0f,#232526)",
    "Galaxy 🌌": "linear-gradient(135deg,#0f2027,#203a43,#2c5364)",
    "Forest 🌿": "linear-gradient(135deg,#134e5e,#71b280)"
}

st.sidebar.title("🎨 Design")
theme_choice = st.sidebar.selectbox("Background Theme", list(themes.keys()))

st.markdown(f"""
<style>
    .stApp, [data-testid="stSidebar"], .stMain {{
        background: {themes[theme_choice]} !important;
    }}
    .metric-card {{
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
</style>
""", unsafe_allow_html=True)

st.title("💳 Credit Risk Assessment Intelligence")
st.sidebar.markdown("---")
model_choice = st.sidebar.radio("🤖 AI Engine", ["Logistic Regression", "Random Forest"])
active_model = lr_model if model_choice == "Logistic Regression" else rf_model

tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "👤 Individual Assessment", "📂 Bulk Data", "ℹ️ About System"])

# =====================================================
# 5. TAB 1: PERFORMANCE
# =====================================================
with tab1:
    y_pred = active_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><h4>Accuracy</h4><h2>{acc*100:.1f}%</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h4>Engine</h4><h2>{model_choice.split()[0]}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h4>Samples</h4><h2>{len(y_test):,}</h2></div>', unsafe_allow_html=True)
    st.write("#### Classification Report")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose(), use_container_width=True)

# =====================================================
# 6. TAB 2: INDIVIDUAL ASSESSMENT
# =====================================================
with tab2:
    st.markdown("### 👤 Applicant Profiling")
    
# Updated instruction block with the specific class
    st.markdown("""
<div class="custom-instruction-box">
    📋 <strong>Instructions:</strong> Fields marked with <span class='required-asterisk'>*</span> are mandatory. Hover over ℹ️ for help.
</div>
""", unsafe_allow_html=True)
    # Clear form functionality
    if 'form_cleared' not in st.session_state:
        st.session_state.form_cleared = False
    
    if st.button("🧹 CLEAR FORM", use_container_width=True):
        st.session_state.form_cleared = True
        try:
            keys_to_clear = []
            for key in st.session_state.keys():
                if isinstance(key, str) and key.startswith("form_"):
                    keys_to_clear.append(key)
            for key in keys_to_clear:
                del st.session_state[key]
        except Exception:
            pass
        st.rerun()
    
    # Main form with compact layout
    with st.form("assessment_form"):
        left, right = st.columns(2)
        u_input = {}

        with left:
            st.markdown("#### 💰 Financial Values")
            
            for col in num_c:
                label = col.replace('_', ' ').title()
                is_required = col in REQUIRED_FIELDS
                
                # Detailed help text for tooltip
                detailed_help = {
                    "loan_amount": "Total money borrowed - the principal amount you're requesting",
                    "rate_of_interest": "Annual interest rate (APR) charged on the loan",
                    "term": "Loan duration in months for repayment",
                    "property_value": "Current market value of property used as collateral",
                    "income": "Gross monthly income from all sources before taxes",
                    "Credit_Score": "FICO score (300-850) indicating creditworthiness",
                    "ltv": "Loan-to-value ratio: (Loan Amount ÷ Property Value) × 100",
                    "dtir": "Debt-to-income ratio: (Total Debt ÷ Income) × 100",
                    "neg_lumpsum": "Any large negative cash flow or outstanding debt",
                    "loan_limit": "Maximum loan amount approved for your profile",
                    "age": "Your current age in years"
                }.get(col, "Financial information")
                
                # Label with help icon tooltip
                if is_required:
                    st.markdown(f'**{label}** <span class="required-asterisk">*</span> <span title="{detailed_help}" style="color: #007BFF; cursor: help;">ℹ️</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'**{label}** <span title="{detailed_help}" style="color: #007BFF; cursor: help;">ℹ️</span>', unsafe_allow_html=True)
                
                # Input field
                if st.session_state.form_cleared:
                    initial_value = 0.0
                else:
                    if is_required:
                        initial_value = st.session_state.get(f"form_{col}", 0.0)
                    else:
                        initial_value = st.session_state.get(f"form_{col}", num_s[col]['med'])
                
                u_input[col] = st.number_input(
                    label="",
                    value=float(initial_value),
                    min_value=0.0,
                    step=0.01,
                    key=f"form_{col}",
                    label_visibility="collapsed"
                )
                
                # Helper text below input
                helper_text = {
                    "loan_amount": "Enter the total loan amount you're requesting",
                    "rate_of_interest": "Annual interest rate as a percentage",
                    "term": "Loan duration in months (e.g., 360 for 30 years)",
                    "property_value": "Current market value of the property",
                    "income": "Your gross monthly income before taxes",
                    "Credit_Score": "Your credit score (300-850 range)",
                    "LTV": "Loan-to-value ratio as a percentage",
                    "dtir1": "Debt-to-income ratio as a percentage",
                    "neg_lumpsum": "Any negative lump sum amount",
                    "loan_limit": "Maximum approved loan amount",
                    "Age": "Your current age in years"
                }.get(col, "Enter the required value")
                
                st.markdown(f'<p style="font-size: 12px; color: #666; margin-top: -8px; margin-bottom: 12px;">{helper_text}</p>', unsafe_allow_html=True)

        with right:
            st.markdown("#### 📋 Qualitative Factors")
            
            for col in cat_c:
                label = col.replace('_', ' ').title()
                is_required = col in REQUIRED_FIELDS
                options = ui_opts[col]
                
                # Detailed help text for tooltip
                detailed_help = {
                    "Gender": "Applicant's gender for demographic analysis",
                    "approv_in_adv": "Pre-approval status: Yes (pre-approved) or No (standard process)",
                    "loan_type": "Type1: Personal Loan, Type2: Home Loan, Type3: Education Loan",
                    "loan_purpose": "P1: Business ventures, P2: Education expenses, P3: Medical needs",
                    "Credit_Worthiness": "L1: Low (poor history), L2: Medium (average), L3: High (excellent)",
                    "submission_of_application": "Application submission and verification method",
                    "credit_type": "CF: Credit File (traditional score), NCF: Non-Credit File (alternative)"
                }.get(col, "Categorical selection")
                
                # Label with help icon tooltip
                if is_required:
                    st.markdown(f'**{label}** <span class="required-asterisk">*</span> <span title="{detailed_help}" style="color: #007BFF; cursor: help;">ℹ️</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'**{label}** <span title="{detailed_help}" style="color: #007BFF; cursor: help;">ℹ️</span>', unsafe_allow_html=True)
                
                # Select options
                if st.session_state.form_cleared and is_required:
                    display_options = ["Select..."] + options
                    default_index = 0
                else:
                    display_options = options
                    default_index = 0
                
                choice_idx = st.selectbox(
                    label="",
                    options=range(len(display_options)),
                    format_func=lambda x: display_options[x],
                    index=default_index,
                    key=f"form_{col}",
                    label_visibility="collapsed"
                )
                
                # Store value
                if st.session_state.form_cleared and is_required and choice_idx == 0:
                    u_input[col] = None
                else:
                    if st.session_state.form_cleared and is_required:
                        actual_idx = choice_idx - 1
                        u_input[col] = options[actual_idx] if actual_idx >= 0 else None
                    else:
                        u_input[col] = options[choice_idx]
                
                # Helper text below input
                helper_text = {
                    "Gender": "Select your gender",
                    "approv_in_adv": "Choose if you have pre-approval (Yes/No)",
                    "loan_type": "(Personal/Home/Education) loan category",
                    "loan_purpose": "(Business/Education/Medical/Other) purpose",
                    "Credit_Worthiness": "(Low/Medium/High) credit history level",
                    "submission_of_application": "Application submission method",
                    "credit_type": "Credit File (CF) or Non-Credit File (NCF)"
                }.get(col, "Make your selection")
                
                st.markdown(f'<p style="font-size: 12px; color: #666; margin-top: -8px; margin-bottom: 12px;">{helper_text}</p>', unsafe_allow_html=True)

        # Reset cleared flag
        if st.session_state.form_cleared:
            st.session_state.form_cleared = False

        st.markdown("### 🚀 Generate Risk Assessment")
        
        # Submit button with validation
        if st.form_submit_button("RUN PREDICTION", use_container_width=True):
            is_valid, error_messages = validate_form_data(u_input)
            
            if not is_valid:
                st.markdown('<div class="validation-error">', unsafe_allow_html=True)
                st.error("**⚠️ Please fix the following issues before submitting:**")
                for error in error_messages:
                    st.markdown(f"• {error}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                try:
                    clean_input = {k: v for k, v in u_input.items() if v is not None}
                    
                    in_df = pd.DataFrame([clean_input])
                    in_enc = pd.get_dummies(in_df).reindex(columns=feat_names, fill_value=0)
                    in_scl = scaler.transform(in_enc)
                    res = active_model.predict(in_scl)[0]
                    prob = active_model.predict_proba(in_scl)[0][1]
                    
                    st.markdown("---")
                    st.success("✅ **Risk Assessment Complete!**")
                    
                    result_left, result_right = st.columns(2)
                    
                    with result_left:
                        if res == 1:
                            st.error("### 🚩 HIGH RISK")
                            st.markdown("**Recommendation:** Requires additional review and stricter terms")
                        else:
                            st.success("### ✅ LOW RISK")
                            st.markdown("**Recommendation:** Eligible for standard loan processing")
                    
                    with result_right:
                        st.metric("Default Probability", f"{prob*100:.1f}%")
                        
                        if prob < 0.3:
                            st.success("🟢 Low Risk Range")
                        elif prob < 0.7:
                            st.warning("🟡 Medium Risk Range")
                        else:
                            st.error("🔴 High Risk Range")
                
                except Exception as e:
                    st.error(f"❌ **Processing Error:** {str(e)}")
                    st.markdown("Please check your input values and try again.")
# 7. TAB 3: BULK DATA
# =====================================================
with tab3:
    st.markdown("### 📂 Bulk Processing")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        bulk_df = pd.read_csv(uploaded)
        if st.button("🚀 RUN BATCH PREDICTION"):
            temp_df = bulk_df.copy()
            temp_df.columns = temp_df.columns.str.strip().str.replace(" ", "_")
            bulk_enc = pd.get_dummies(temp_df).reindex(columns=feat_names, fill_value=0)
            bulk_scaled = scaler.transform(bulk_enc)
            bulk_df['Risk_Status'] = ["HIGH RISK" if p == 1 else "LOW RISK" for p in active_model.predict(bulk_scaled)]
            st.dataframe(bulk_df.head())

# =====================================================
# 8. TAB 4: ABOUT (Informational Only)
# =====================================================
with tab4:
    st.markdown("## ℹ️ About Credit Risk AI")
    
    st.markdown("### 🎯 Project Purpose")
    st.info("""
    This tool helps lenders decide if a loan application is safe or risky. 
    In the real world, this speeds up the approval process and helps banks 
    avoid lending to situations where the money might not be paid back.
    """)

    st.markdown("### ⚙️ How the System Works")
    st.info("""
    1. **Data Collection:** The system takes in financial details (Income, Loan Amount) and history (Credit Score).
    2. **Comparison:** It compares the new application against thousands of historical loan outcomes.
    3. **Evaluation:** The AI looks for patterns—like high debt compared to low income—that usually lead to defaults.
    4. **Recommendation:** Within seconds, it provides a 'High' or 'Low' risk rating.
    """)

    st.markdown("### 📝 Field Explanations")
    c1, c2 = st.columns(2)
    with c1:
        st.info("""
        * **Loan Amount:** The total principal requested by the applicant.
        * **Rate of Interest:** The annual percentage rate (APR) for the loan.
        * **Term:** The repayment duration in total months.
        * **Property Value:** The market valuation of the collateral property.
        """)
    with c2:
        st.info("""
        * **Income:** Total gross monthly earnings before tax deductions.
        * **Credit Score:** A numerical expression (300-850) of creditworthiness.
        * **LTV (Loan-to-Value):** Ratio of the loan amount to the property value.
        * **DTIR (Debt-to-Income):** Percentage of monthly income used for debt.
        """)

    st.markdown("### 🧹 Why some data was removed")
    st.warning("""
    We removed fields like **ID numbers, Names, and Dates**. These are 'noisy' 
    because they don't help predict risk. Knowing a name doesn't tell 
    the AI if someone can pay a loan, so we remove them for fairness and accuracy.
    """)
    # --- Qualitative & Risk Factors Section ---
    st.markdown("### 📋 Qualitative & Risk Factors")
    col_qual1, col_qual2 = st.columns(2)
    with col_qual1:
        st.info("""
        * **Loan Limit:** The maximum approved ceiling for the applicant's profile.
        * **Gender:** Demographic information for regulatory and statistical use.
        * **Approv In Adv:** Indicates if the user had a pre-approval status.
        * **Loan Type/Purpose:** Category (Personal/Home) and the reason for borrowing.
        * **Credit Worthiness:** Internal assessment of past payment reliability.
        * **Age:** Categorical age bracket of the applicant.
        """)
    with col_qual2:
        st.info("""
        * **Business or Commercial:** Indicates if the loan is for a business entity.
        * **Neg Ammortization:** If the loan balance increases despite making payments.
        * **Interest Only:** Payments that only cover interest, not principal.
        * **Lump Sum Payment:** Provision for a single large final payment.
        * **Secured By:** The type of asset backing the loan (e.g., Home).
        * **Credit Type:** Traditional (CF) vs. Alternative (NCF) credit data.
        """)
    st.markdown("### 🚦 Interpreting Results")
    st.info("""
    * **✅ LOW RISK:** Application is statistically safe based on past data.
    * **🚩 HIGH RISK:** Application matches patterns of past loan defaults.
    * **Probability:** Closer to 100% means higher certainty of potential default.
    """)

    st.markdown("---")
    st.caption("Developed as an AI-driven decision support tool.")
