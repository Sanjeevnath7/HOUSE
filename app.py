import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import joblib
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.neighbors import NearestNeighbors

def load_and_encode_data(path="data/house2.csv"):
    df = pd.read_csv(path)

    # Handle nulls
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode categorical features
    cat_cols = ["location"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders



# Train & Save Models
def train_models(df):
    X = df[["sqft", "bedrooms", "bathrooms", "location_encoded"]]
    y = df["price"]

    # Linear Regression
    lr = LinearRegression().fit(X, y)
    pickle.dump(lr, open("models/linear_model.pkl", "wb"))

    # XGBoost
    xgb = XGBRegressor().fit(X, y)
    pickle.dump(xgb, open("models/xgb_model.pkl", "wb"))

    # SARIMA / SARIMAX (time series)
    ts = df.groupby("date")["price"].mean()
    sarimax = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
    pickle.dump(sarimax, open("models/sarimax_model.pkl", "wb"))

def load_models():
    models = {}
    for m in ["linear_model.pkl", "xgb_model.pkl", "sarimax_model.pkl"]:
        try:
            models[m] = pickle.load(open("models/" + m, "rb"))
        except:
            models[m] = None
    return models

def predict(models, new_house):
    X = pd.DataFrame([new_house])

    preds = []
    if models["linear_model.pkl"]:
        preds.append(models["linear_model.pkl"].predict(X)[0])
    if models["xgb_model.pkl"]:
        preds.append(models["xgb_model.pkl"].predict(X)[0])
    if models["sarimax_model.pkl"]:
        preds.append(models["sarimax_model.pkl"].forecast(steps=1)[0])

    return np.mean(preds)



def load_models():
    """
    Load trained ML models (Linear Regression, XGBoost, etc.)
    """
    models = {}
    try:
        models["linear"] = joblib.load("linear_regression.pkl")
        models["xgb"] = joblib.load("xgboost.pkl")
        # Add more if you saved SARIMA / SARIMAX
    except Exception as e:
        print("⚠ Error loading models:", e)
    return models

def predict(models, house_features: dict):
    """
    Predict house price using available models.
    Here we average predictions across all models.
    """
    X = [[
        house_features["sqft"],
        house_features["bedrooms"],
        house_features["bathrooms"],
        house_features["location_encoded"]
    ]]

    preds = []
    for name, model in models.items():
        try:
            preds.append(model.predict(X)[0])
        except Exception as e:
            print(f"⚠ Model {name} failed:", e)

    return sum(preds) / len(preds) if preds else 0



def comparative_market_analysis(new_house, df, k=10):
    features = ["sqft", "bedrooms", "bathrooms", "location_encoded"]
    knn = NearestNeighbors(n_neighbors=k).fit(df[features])

    new_df = pd.DataFrame([new_house])
    distances, indices = knn.kneighbors(new_df)
    similar = df.iloc[indices[0]]

    return {
        "low": similar["price"].quantile(0.25),
        "median": similar["price"].median(),
        "high": similar["price"].quantile(0.75),
        "similar": similar
    }



def load_and_encode_data():
    """
    Load dataset and encoders
    """
    try:
        df = pd.read_csv("house2.csv")
        encoders = joblib.load("encoders.pkl")   # LabelEncoders, etc.
    except Exception as e:
        print("⚠ Error loading data/encoders:", e)
        df = pd.DataFrame()
        encoders = {}
    return df, encoders

def comparative_market_analysis(house, df):
    """
    Compare input house with similar properties in dataset.
    """
    if df.empty:
        return {"low": 0, "high": 0, "median": 0}

    # Filter by location + bedrooms
    similar = df[
        (df["location_encoded"] == house["location_encoded"]) &
        (df["bedrooms"] == house["bedrooms"])
    ]

    if similar.empty:
        return {"low": 0, "high": 0, "median": 0}

    prices = similar["price"].values
    return {
        "low": np.min(prices),
        "high": np.max(prices),
        "median": np.median(prices)
    }



def load_and_encode_data():
    """
    Load dataset and encoders
    """
    try:
        df = pd.read_csv("house2.csv")
        encoders = joblib.load("encoders.pkl")   # LabelEncoders, etc.
    except Exception as e:
        print("⚠ Error loading data/encoders:", e)
        df = pd.DataFrame()
        encoders = {}
    return df, encoders

def comparative_market_analysis(house, df):
    """
    Compare input house with similar properties in dataset.
    """
    if df.empty:
        return {"low": 0, "high": 0, "median": 0}

    # Filter by location + bedrooms
    similar = df[
        (df["location_encoded"] == house["location_encoded"]) &
        (df["bedrooms"] == house["bedrooms"])
    ]

    if similar.empty:
        return {"low": 0, "high": 0, "median": 0}

    prices = similar["price"].values
    return {
        "low": np.min(prices),
        "high": np.max(prices),
        "median": np.median(prices)
    }



# --- Load Models & Encoders ---
@st.cache_resource
def load_models():
    models = {}
    try:
        models["linear"] = joblib.load("linear_regression.pkl")
        models["xgb"] = joblib.load("xgboost.pkl")
    except Exception as e:
        st.error(f"⚠ Error loading models: {e}")
    return models

@st.cache_data
def load_and_encode_data():
    try:
        df = pd.read_csv("housing_data.csv")
        encoders = joblib.load("encoders.pkl")
    except Exception as e:
        st.error(f"⚠ Error loading data/encoders: {e}")
        df, encoders = pd.DataFrame(), {}
    return df, encoders

def predict(models, house_features: dict):
    X = [[
        house_features["sqft"],
        house_features["bedrooms"],
        house_features["bathrooms"],
        house_features["location_encoded"]
    ]]
    preds = []
    for name, model in models.items():
        try:
            preds.append(model.predict(X)[0])
        except Exception as e:
            st.warning(f"Model {name} failed: {e}")
    return sum(preds) / len(preds) if preds else 0

def comparative_market_analysis(house, df):
    if df.empty:
        return {"low": 0, "high": 0, "median": 0}
    similar = df[
        (df["location_encoded"] == house["location_encoded"]) &
        (df["bedrooms"] == house["bedrooms"])
    ]
    if similar.empty:
        return {"low": 0, "high": 0, "median": 0}
    prices = similar["price"].values
    return {"low": np.min(prices), "high": np.max(prices), "median": np.median(prices)}

# --- Streamlit UI ---
st.set_page_config(page_title="🏠 House Price Chatbot", layout="centered")
st.title("🏠 House Price Prediction Chatbot")

df, encoders = load_and_encode_data()
models = load_models()

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show previous messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me about house prices..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Extract structured info (simple demo: ask directly)
    sqft = st.number_input("📐 Enter square feet:", min_value=500, max_value=10000, step=50)
    bedrooms = st.number_input("🛏 Bedrooms:", min_value=1, max_value=10, step=1)
    location = st.text_input("📍 Location:")

    if st.button("Predict"):
        try:
            loc_encoded = encoders["location"].transform([location])[0]
            house = {"sqft": sqft, "bedrooms": bedrooms, "bathrooms": 2, "location_encoded": loc_encoded}
            price = predict(models, house)
            cma = comparative_market_analysis(house, df)

            response = (
                f"🏠 Predicted: ₹{price:,.0f}\n\n"
                f"📊 CMA Range: ₹{cma['low']:,.0f} - ₹{cma['high']:,.0f}\n\n"
                f"💡 Median: ₹{cma['median']:,.0f}"
            )
        except Exception as e:
            response = f"⚠ Error: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)