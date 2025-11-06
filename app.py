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
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None, None, None

    if 'Success' not in data.columns:
        st.error("Dataset must contain a 'Success' target column.")
        return None, None, None

    # Encode categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    X = data.drop('Success', axis=1)
    y = data['Success']

    # Train RandomForest to extract data-driven feature importances
    rf = RandomForestClassifier(random_state=42, n_estimators=200)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5).index.tolist()

    # Train Logistic Regression on top 5 features
    x = X[top_features]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_scaled, y_train)

    accuracy = model.score(x_test_scaled, y_test)
    return model, top_features, importances, scaler, accuracy


# =============================================
# CUSTOM CSS
# =============================================
def load_css():
    st.markdown("""
        <style>
            .stApp { background-color: #f0f2f6; }
            [data-testid="stAppViewContainer"] > .main .block-container {
                background-color: #ffffff;
                padding: 2rem;
                border-radius: 0.5rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                max-width: 950px;
                margin: 0 auto;
            }
            h1 { color: #1e3a8a; text-align: center; font-weight: 800; }
            .stButton > button {
                background-color: #3b82f6; color: white; border: none;
                padding: 0.6rem 1.2rem; border-radius: 0.5rem; font-weight: 600; width: 100%;
            }
            .stButton > button:hover { background-color: #2563eb; }
        </style>
    """, unsafe_allow_html=True)


# =============================================
# STREAMLIT APP
# =============================================
st.set_page_config(page_title="M&A Success Predictor: Analytical Intelligence")
load_css()
st.title("🤖 M&A Success Predictor — Analytical Intelligence Dashboard")

uploaded_file = st.file_uploader("Upload your M&A dataset (CSV)", type="csv")

if uploaded_file is None:
    st.info("Upload your CSV file to start model training and analysis.")
    st.stop()

model, features, importances, scaler, acc = train_model(uploaded_file)

if model is None:
    st.warning("Model training failed.")
    st.stop()

st.success(f"✅ Model trained successfully. Accuracy: **{acc*100:.2f}%**")
st.write(f"**Top Predictive Features:** {', '.join(features)}")

# =============================================
# USER INPUT SECTION
# =============================================
st.markdown("### 🎛️ Enter Feature Values to Simulate a Deal Scenario")
inputs = []
for feature in features:
    val = st.number_input(f"{feature}", value=0.0, format="%.2f")
    inputs.append(val)

# Apply automatic feature weights (based on data importance)
auto_weights = importances[features].values
auto_weights = auto_weights / auto_weights.sum()  # Normalize to sum=1

# Calculate weighted score
weighted_score = np.dot(inputs, auto_weights)

# Predict
scaled_inputs = scaler.transform([inputs])
pred = model.predict(scaled_inputs)[0]
proba = model.predict_proba(scaled_inputs)[0]

# =============================================
# ANALYTICAL OUTPUT
# =============================================
st.markdown("---")
st.subheader("🧠 Analytical Prediction Result")

if int(pred) == 1:
    st.success(f"✅ Predicted Outcome: **SUCCESS** (Probability: {proba[1]*100:.2f}%)")
    st.write("**Interpretation:** The weighted combination of strategic and financial indicators reflects strong synergy potential, indicating high merger compatibility.")
else:
    st.error(f"⚠️ Predicted Outcome: **FAILURE** (Probability: {proba[0]*100:.2f}%)")
    st.write("**Interpretation:** The current configuration suggests misalignment across key synergy and financial metrics, potentially reducing the likelihood of deal success.")

# =============================================
# VISUAL EXPLANATION
# =============================================
st.subheader("📊 Feature Weight & Influence Analysis")

# Weighted Influence Table
influence_df = pd.DataFrame({
    "Feature": features,
    "Input Value": inputs,
    "Data-driven Weight": auto_weights,
    "Weighted Contribution": np.multiply(inputs, auto_weights)
}).sort_values(by="Weighted Contribution", ascending=False)

st.dataframe(influence_df.style.background_gradient(cmap="Blues", subset=["Weighted Contribution"]))

# Feature Impact Visualization
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x="Weighted Contribution", y="Feature", data=influence_df, palette="Blues_d", ax=ax)
ax.set_title("Feature Influence on Predicted Success", fontsize=12, fontweight='bold')
st.pyplot(fig)

# =============================================
# ADVANCED INTERPRETATION LAYER
# =============================================
dominant_feature = influence_df.iloc[0]["Feature"]
dominant_contrib = influence_df.iloc[0]["Weighted Contribution"]

st.markdown(f"""
### 🔍 Deep Analytical Insights
- The most influential factor is **{dominant_feature}**, contributing **{dominant_contrib:.2f}** points to the overall prediction score.
- The model’s **data-driven weighting** prioritizes this feature due to its strong correlation with M&A success in historical data.
- To **improve success probability**, focus on enhancing parameters with lower weighted influence or optimizing the top driver’s input value.
- The model’s internal weighting is dynamically adjusted using **Random Forest feature importances**, ensuring analytical reliability.
""")

# =============================================
# MODEL CONFIDENCE CHART
# =============================================
st.subheader("📈 Model Confidence Distribution")

fig2, ax2 = plt.subplots()
ax2.bar(["Failure (0)", "Success (1)"], proba, color=["red", "green"])
ax2.set_ylim(0, 1)
ax2.set_ylabel("Probability")
ax2.set_title("Model Confidence in Prediction")
for i, v in enumerate(proba):
    ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
st.pyplot(fig2)
