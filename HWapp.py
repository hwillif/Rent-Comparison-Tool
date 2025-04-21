import pandas as pd
import numpy as np
import streamlit as st


# Page Title
st.title('Rent Comparison Tool')
st.markdown("Hayden Williford, Matthew Rostar, Jack Callahan, Lillian Bowling")

st.header("Your Apartment Details:")
col1, col2, col3 = st.columns(3)
with col1:
    user_rent = st.text_input("Rent Amount? (Dollars)")
    if user_rent:
        st.markdown("Rent:", value(user_rent))

with col2:
    user_bedrooms = st.text_input("Number of Bedrooms?")
    if user_bedrooms:
        st.markdown("Bedrooms:", value(user_bedrooms))

with col3:
    user_bathrooms = st.text_input("Number of Bathrooms?")
    if user_bathrooms:
        st.markdown("Bathrooms:", value(user_bathrooms))


