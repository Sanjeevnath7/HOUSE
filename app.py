# real_estate_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import NearestNeighbors
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------------
# 1. Load / Prepare Data
# ----------------------------
@st.cache_data
def load_data(path="home2.csv"):
    df = pd.read_csv(path)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

df = load_data()

# Encode categorical features
if "location_encoded" not in df.columns:
    le_location = LabelEncoder()
    df["location_encoded"] = le_location.fit_transform(df["City"])
else:
    le_location = LabelEncoder()
    le_location.fit(df["City"])

# ----------------------------
# 2. Train / Load Models
# ----------------------------
# Linear Regression
if "linear_model.pkl" in df.columns:
    try:
        lr_model = pickle.load(open("linear_model.pkl", "rb"))
    except:
        lr_model = LinearRegression()
        X = df[["Area", "Bedrooms", "Bathrooms", "location_encoded"]]
        y = df["Price"]
        lr_model.fit(X, y)
        pickle.dump(lr_model, open("linear_model.pkl", "wb"))
else:
    lr_model = LinearRegression()
    X = df[["Area", "Bedrooms", "Bathrooms", "location_encoded"]]
    y = df["Price"]
    lr_model.fit(X, y)
    pickle.dump(lr_model, open("linear_model.pkl", "wb"))

# XGBoost
try:
    xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
except:
    xgb_model = XGBRegressor()
    xgb_model.fit(X, y)
    pickle.dump(xgb_model, open("xgb_model.pkl", "wb"))

# SARIMAX
try:
    sarimax_model = pickle.load(open("sarimax_model.pkl", "rb"))
except:
    ts = df.groupby("Year")["Price"].mean() if "Year" in df.columns else df["Price"]
    sarimax_model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12) if "Year" in df.columns else (0,0,0,0)).fit(disp=False)
    pickle.dump(sarimax_model, open("sarimax_model.pkl", "wb"))

# ----------------------------
# 3. Helper Functions
# ----------------------------
def preprocess_input(new_house):
    return pd.DataFrame([new_house])

def predict_price(new_house):
    X_new = preprocess_input(new_house)
    preds = []
    if lr_model:
        preds.append(float(lr_model.predict(X_new)[0]))
    if xgb_model:
        preds.append(float(xgb_model.predict(X_new)[0]))
    if sarimax_model:
        try:
            preds.append(float(sarimax_model.forecast(steps=1)[0]))
        except:
            pass
    return np.mean(preds) if preds else 5000000

def comparative_market_analysis(new_house, k=10):
    features = ["Area", "Bedrooms", "Bathrooms", "location_encoded"]
    knn = NearestNeighbors(n_neighbors=k).fit(df[features])
    X_new = pd.DataFrame([new_house])
    distances, indices = knn.kneighbors(X_new)
    similar = df.iloc[indices[0]]
    return similar, {
        "low": similar["price"].quantile(0.25),
        "median": similar["price"].median(),
        "high": similar["price"].quantile(0.75)
    }

def parse_chat_input(text):
    sqft = re.search(r"(\d+)\s*Area", text)
    bhk = re.search(r"(\d+)\s*BHK", text, re.IGNORECASE)
    loc = re.search(r"in\s+([a-zA-Z\s]+)", text)
    return {
        "sqft": int(sqft.group(1)) if sqft else 1000,
        "bedrooms": int(bhk.group(1)) if bhk else 2,
        "bathrooms": 2,
        "location_encoded": le_location.transform([loc.group(1)])[0] if loc else 0
    }

def chatbot_reply(user_text):
    house = parse_chat_input(user_text)
    price = predict_price(house)
    similar, stats = comparative_market_analysis(house)
    reply = f"🏠 Predicted Price: ₹{price:,.0f}\n"
    reply += f"📊 Market Range: ₹{stats['low']:,.0f} - ₹{stats['high']:,.0f}\n"
    reply += f"💡 Median Price: ₹{stats['median']:,.0f}\n\n"
    reply += "Similar Houses:\n"
    reply += similar[["price","sqft","bedrooms"]].head(5).to_string(index=False)
    return reply

# ----------------------------
# 4. Streamlit UI
# ----------------------------
st.set_page_config(page_title="🏠 Real Estate Assistant", layout="wide")
st.title("🏠 Real Estate Price Assistant")

tab1, tab2, tab3 = st.tabs(["🔮 Price Prediction","📊 CMA","💬 Chatbot"])

with tab1:
    st.header("Price Prediction")
    sqft = st.number_input("Square Feet", 500, 5000, 1200)
    bedrooms = st.number_input("Bedrooms", 1, 10, 2)
    bathrooms = st.number_input("Bathrooms", 1, 5, 2)
    location = st.selectbox("City", df["City"].unique())
    loc_encoded = le_location.transform([location])[0]

    if st.button("Predict Price"):
        house = {"sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms, "location_encoded": loc_encoded}
        price = predict_price(house)
        st.success(f"Estimated Price: ₹{price:,.0f}")

with tab2:
    st.header("Comparative Market Analysis")
    sqft = st.number_input("Square Feet (CMA)", 500, 5000, 1500)
    bedrooms = st.number_input("Bedrooms (CMA)", 1, 10, 2)
    bathrooms = st.number_input("Bathrooms (CMA)", 1, 5, 2)
    location = st.selectbox("City (CMA)", df["City"].unique())
    loc_encoded = le_location.transform([location])[0]

    if st.button("Run CMA"):
        house = {"sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms, "location_encoded": loc_encoded}
        similar, stats = comparative_market_analysis(house)
        st.write(f"📊 Price Range: ₹{stats['low']:,.0f} - ₹{stats['high']:,.0f}")
        st.write(f"💡 Median: ₹{stats['median']:,.0f}")
        st.dataframe(similar[["price","sqft","bedrooms","bathrooms","location"]])

with tab3:
    st.header("Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask your query (e.g., 'Price of 2BHK in Mumbai 1200 sqft')")
    if st.button("Send"):
        st.session_state.chat_history.append(("You", user_input))
        reply = chatbot_reply(user_input)
        st.session_state.chat_history.append(("Bot", reply))

    for role, msg in st.session_state.chat_history:
        st.markdown(f"{role}:** {msg}")