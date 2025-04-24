import pandas as pd
import streamlit as st 
import numpy as np
from sklearn.linear_model import LinearRegression


# map
map_data = pd.DataFrame({
    'lat': np.random.uniform(40.70, 40.85, 100),
    'lon': np.random.uniform(-74.02, -73.93, 100)
})

st.title("Map Example")

st.map(map_data)

df = pd.read_csv("lean_apartments.csv")  # Ensure columns are named 'lat' and 'lon'
st.map(df)


@st.cache_data
def load_data():
    df = pd.read_csv("clean_apartments.csv")
    return df

df = load_data()

feature_cols = ['bedrooms', 'bathrooms', 'square_feet', 'pets?','latitude','longitude']


#linreg
@st.cache_data
def train_regression_model(cluster_df):
    cluster_df = cluster_df.dropna(subset=['price'])
    
    X = cluster_df[feature_cols]
    y = cluster_df['price']
    
    model = LinearRegression()
    model.fit(X, y)
    return model