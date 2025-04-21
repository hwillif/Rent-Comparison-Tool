import pandas as pd
import numpy as np
import streamlit as st


# Page Title
st.title('Rent Comparison Tool')
st.markdown("Hayden Williford, Matthew Rostar, Jack Callahan, Lillian Bowling")

st.header("Your Apartment Details:")

col1, col2, col3 = st.columns(3)
with col1:
    # User Rent Text Box
    user_rent = st.text_input("Rent Amount?")

    # Display the Rent value and restrict text to numbers
    if user_rent:
        if user_rent.isdigit():
            number = int(user_rent)
            st.write(f"Rent Amount: ${number}")
        else:
            st.error("Please enter a valid number")

    #Pets Allowed Drop Down
    user_pets = st.selectbox("Pets Allowed?", 
                             ("No","Yes"), 
                             index = None)
    if user_pets != None:
        st.write(f"Pets Allowed: {user_pets}")


with col2:
    # User Bedrooms Text Box
    user_bedrooms = st.text_input("Number of Bedrooms?")

    # Display the Bedrooms value and restrict text to numbers
    if user_bedrooms:
        if user_bedrooms.isdigit():
            number = int(user_bedrooms)
            st.write(f"Number of Bedrooms: {number}")
        else:
            st.error("Please enter a valid number")

with col3:
    # User Bathrooms Text Box
    user_bathrooms = st.text_input("Number of Bathrooms?")

    # Display the Number of Bathrooms value and restrict values
    if user_bathrooms :
        try:
            value = float(user_bathrooms)
            if 0 <= value <= 10 and (value * 2).is_integer():
                st.write(f"Number of Bedrooms: {value}")
            else:
                st.error("Number must be between 0 and 10, in 0.5 steps.")
        except ValueError:
            st.error("Please enter a valid number.")
        
