import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# =============================================
# MODEL TRAINING FUNCTION
# =============================================
@st.cache_resource
def train_model(uploaded_file):
    """
    Reads dataset, encodes categorical variables, ranks features by importance,
    and trains a weighted logistic regression model for prediction.
    """
    try:
        # Load the uploaded CSV file into a DataFrame
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None, None, None, None, None

    # Check for the mandatory target column 'Success'
    if 'Success' not in data.columns:
        st.error("Dataset must contain a 'Success' target column (0 for Failure, 1 for Success).")
        return None, None, None, None, None

    # --- Preprocessing ---
    # Encode categorical (object) columns using LabelEncoder
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        # Convert column to string type before fitting to handle mixed types gracefully
        data[col] = le.fit_transform(data[col].astype(str))

    X = data.drop('Success', axis=1)
    y = data['Success']

    # --- Feature Importance using Random Forest ---
    # Train RandomForest to extract data-driven feature importances
    rf = RandomForestClassifier(random_state=42, n_estimators=200)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    # Select the top 5 most important features
    top_features = importances.sort_values(ascending=False).head(5).index.tolist()

    # --- Logistic Regression Training ---
    # Use only the top 5 features for the final model
    x = X[top_features]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_scaled, y_train)

    # Evaluate model accuracy
    accuracy = model.score(x_test_scaled, y_test)

    # Return trained model, top features, full importances, scaler, and accuracy
    return model, top_features, importances, scaler, accuracy


# =============================================
# CUSTOM CSS for Aesthetic Styling
# =============================================
def load_css():
    """Injects custom CSS for a modern, clean Streamlit design."""
    st.markdown("""
        <style>
            .stApp { background-color: #f0f2f6; }
            [data-testid="stAppViewContainer"] > .main .block-container {
                background-color: #ffffff;
                padding: 2rem;
                border-radius: 0.75rem; /* Slightly larger radius */
                box-shadow: 0 8px 15px rgba(0,0,0,0.1); /* Deeper shadow */
                max-width: 950px;
                margin: 0 auto;
            }
            h1 { color: #1e3a8a; text-align: center; font-weight: 800; }
            h3 { color: #3b82f6; border-bottom: 2px solid #e0f2fe; padding-bottom: 0.5rem; }
            .stButton > button {
                background-color: #3b82f6; color: white; border: none;
                padding: 0.6rem 1.2rem; border-radius: 0.5rem; font-weight: 600; width: 100%;
                transition: background-color 0.3s ease;
            }
            .stButton > button:hover { background-color: #2563eb; }
        </style>
    """, unsafe_allow_html=True)


# =============================================
# STREAMLIT APP LAYOUT & LOGIC
# =============================================
# Configure page settings
st.set_page_config(
    page_title="M&A Success Predictor: Analytical Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom styling
load_css()
st.title("🤖 M&A Success Predictor — Analytical Intelligence Dashboard")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your M&A dataset (CSV) containing a 'Success' target column (0 or 1)", type="csv")

if uploaded_file is None:
    st.info("Upload your CSV file to start model training and analysis. (Example features: RevenueSynergy, CulturalGap, IntegrationCost, TargetValuation, BuyerDebt)")
    st.stop()

# --- Model Training ---
model, features, importances, scaler, acc = train_model(uploaded_file)

if model is None:
    st.warning("Model training failed. Please check your data format and ensure the 'Success' column exists.")
    st.stop()

st.success(f"✅ Model trained successfully on historical data. Accuracy: **{acc*100:.2f}%**")
st.markdown(f"**Top 5 Predictive Features identified:** `{', '.join(features)}`")

# =============================================
# USER INPUT SECTION
# =============================================
st.markdown("### 🎛️ Enter Feature Values to Simulate a Deal Scenario")
st.markdown("*(Note: Input values will be standardized by the model before prediction)*")

inputs = []
# Create input widgets for the top features identified by the model
cols = st.columns(len(features))
for i, feature in enumerate(features):
    with cols[i]:
        val = st.number_input(f"**{feature}**", value=0.0, format="%.2f", key=f"input_{feature}")
        inputs.append(val)

# Apply automatic feature weights (based on data importance)
auto_weights = importances[features].values
# Normalize weights so they sum up to 1 for easier interpretation of contribution
auto_weights = auto_weights / auto_weights.sum()

# Reshape inputs for the scaler (required by scikit-learn for single sample)
input_data = np.array(inputs).reshape(1, -1)

# Scale inputs using the scaler fitted during training
scaled_inputs = scaler.transform(input_data)

# Predict the outcome and probabilities
pred = model.predict(scaled_inputs)[0]
proba = model.predict_proba(scaled_inputs)[0]

# =============================================
# ANALYTICAL OUTPUT
# =============================================
st.markdown("---")
st.subheader("🧠 Analytical Prediction Result")

col_result, col_weights = st.columns([1, 1])

with col_result:
    if int(pred) == 1:
        st.success(f"## ✅ Predicted Outcome: SUCCESS")
        st.markdown(f"**Probability of Success:** `{proba[1]*100:.2f}%`")
        st.write("**Interpretation:** The weighted combination of strategic and financial indicators reflects strong synergy potential, indicating high merger compatibility for the input scenario.")
    else:
        st.error(f"## ⚠️ Predicted Outcome: FAILURE")
        st.markdown(f"**Probability of Failure:** `{proba[0]*100:.2f}%`")
        st.write("**Interpretation:** The current configuration suggests misalignment across key synergy and financial metrics, potentially reducing the likelihood of deal success.")

# =============================================
# VISUAL EXPLANATION
# =============================================

st.subheader("📊 Feature Weight & Influence Analysis")

# Weighted Influence Table
influence_df = pd.DataFrame({
    "Feature": features,
    "Input Value": inputs,
    "Data-driven Weight (RF)": auto_weights,
    "Weighted Contribution (Input * Weight)": np.multiply(inputs, auto_weights)
}).sort_values(by="Weighted Contribution (Input * Weight)", ascending=False)

st.dataframe(
    influence_df.style.background_gradient(
        cmap="Blues", 
        subset=["Weighted Contribution (Input * Weight)"]
    ).format({
        "Input Value": "{:.2f}",
        "Data-driven Weight (RF)": "{:.3f}",
        "Weighted Contribution (Input * Weight)": "{:.3f}"
    }),
    use_container_width=True
)

# Feature Impact Visualization (Bar Chart)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x="Weighted Contribution (Input * Weight)", y="Feature", data=influence_df, palette="Blues_d", ax=ax)
ax.set_title("Feature Influence on Prediction Score (Higher Contribution = More Influence)", fontsize=12, fontweight='bold')
ax.set_xlabel("Weighted Contribution Score")
st.pyplot(fig)


# =============================================
# ADVANCED INTERPRETATION LAYER
# =============================================
dominant_feature = influence_df.iloc[0]["Feature"]
dominant_contrib = influence_df.iloc[0]["Weighted Contribution (Input * Weight)"]

st.markdown(f"""
### 🔍 Deep Analytical Insights
- The most influential factor in the current scenario is **{dominant_feature}**, which has the highest **Weighted Contribution** score of **{dominant_contrib:.3f}**.
- The model’s **data-driven weighting** (derived from Random Forest) prioritizes this feature because of its strong historical correlation with M&A success.
- **Actionable Insight:** To **improve success probability** for this deal configuration, focus on enhancing the input value for `{dominant_feature}` or optimizing other parameters with low weighted influence.
""")

# =============================================
# MODEL CONFIDENCE CHART
# =============================================
st.subheader("📈 Model Confidence Distribution")

fig2, ax2 = plt.subplots(figsize=(6, 3))
bar_labels = ["Failure (0)", "Success (1)"]
ax2.bar(bar_labels, proba, color=["#ef4444", "#10b981"])
ax2.set_ylim(0, 1)
ax2.set_ylabel("Prediction Probability")
ax2.set_title("Model Confidence in Outcome")
for i, v in enumerate(proba):
    ax2.text(i, v + 0.05, f"{v*100:.2f}%", ha="center", fontweight="bold")
st.pyplot(fig2)
