import pandas as pd
import numpy as np
import streamlit as st

st.title("Text Input Example")

# Text input field
user_input = st.text_input("Enter some text:")

# Display the entered value
if user_input:
    st.write("You entered:", user_input)
