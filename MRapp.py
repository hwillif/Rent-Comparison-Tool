import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

@st.cache_data
def load_data():
    df = pd.read_csv("clean_apartments.csv")
    return df

df = load_data()

feature_cols = ['bedrooms', 'bathrooms', 'square_feet', 'pets?', 'latitude', 'longitude']

#kmeans
@st.cache_data
def cluster_data(df, n_clusters=50):
    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['cluster'] = clusters
    return df, kmeans, scaler

#linreg
@st.cache_data
def train_regression_model(cluster_df):
    cluster_df = cluster_df.dropna(subset=['price'])
    
    X = cluster_df[feature_cols]
    y = cluster_df['price']
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    return model


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

