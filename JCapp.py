import pandas as pd
import streamlit as st 
import numpy as np


map_data = pd.DataFrame({
    'lat': np.random.uniform(40.70, 40.85, 100),
    'lon': np.random.uniform(-74.02, -73.93, 100)
})

st.title("Map Example")

st.map(map_data)

df = pd.read_csv("lean_apartments.csv")  # Ensure columns are named 'lat' and 'lon'
st.map(df)
