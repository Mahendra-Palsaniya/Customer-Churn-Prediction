import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Try importing XGBoost (optional)
try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception:
    xgb_available = False

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Churn_Modelling.csv")

# Drop unnecessary columns
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Model Selection
# -----------------------------
st.sidebar.title("⚙️ Model Selection")
model_option = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest"] + (["XGBoost"] if xgb_available else [])
)

# Train selected model
if model_option == "Logistic Regression":
    model = LogisticRegression()
elif model_option == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif model_option == "XGBoost" and xgb_available:
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

model.fit(X_train, y_train)

# Predictions for evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏦 Customer Churn Prediction App")

st.write("### 📊 Model Performance")
st.write(f"**Accuracy:** {acc:.2f}")

st.write("**Confusion Matrix:**")
st.write(cm)

st.write("---")
st.write("### 🔍 Enter Customer Details")

credit_score = st.slider("Credit Score", 300, 900, 600)
age = st.slider("Age", 18, 92, 30)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", value=50000.0)
num_products = st.slider("Number of Products", 1, 4, 1)
has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
est_salary = st.number_input("Estimated Salary", value=50000.0)

# Geography & Gender
geography_germany = st.selectbox("Geography: Germany", [0, 1])
geography_spain = st.selectbox("Geography: Spain", [0, 1])
gender_male = st.selectbox("Gender: Male", [0, 1])

# Prediction
if st.button("Predict"):
    input_data = np.array([[credit_score, age, tenure, balance, num_products,
                            has_card, is_active, est_salary,
                            geography_germany, geography_spain, gender_male]])

    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    # 👉 ADD THIS
    prob = model.predict_proba(input_data)
    st.write(f"Churn Probability: {prob[0][1]:.2f}")

    if prediction[0] == 1:
        st.error("❌ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")