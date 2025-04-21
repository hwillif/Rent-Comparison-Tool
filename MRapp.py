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

feature_cols = ['bedrooms', 'bathrooms', 'square_feet', 'pets?','latitude','longitude']

#kmeans
@st.cache_data
def cluster_data(df, n_clusters=5):
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
    
    model = LinearRegression()
    model.fit(X, y)
    return model


st.title('Rent Comparison Tool')
st.markdown("Hayden Williford, Matthew Rostar, Jack Callahan, Lillian Bowling")

st.header("Your Apartment Details:")
col1, col2, col3 = st.columns(3)

with col1:
    user_rent = st.text_input("Rent Amount?")
    if user_rent.isdigit():
        user_rent = int(user_rent)
    else:
        user_rent = None

    user_pets = st.selectbox("Pets Allowed?", ("No", "Yes"), index=None)

with col2:
    user_bedrooms = st.text_input("Number of Bedrooms?")
    if user_bedrooms.isdigit():
        user_bedrooms = int(user_bedrooms)
    else:
        user_bedrooms = None

with col3:
    user_bathrooms = st.text_input("Number of Bathrooms?")
    try:
        user_bathrooms = float(user_bathrooms)
    except ValueError:
        user_bathrooms = None

#what happens when user clicks on the predict rent button
if st.button("Predict Rent"):

    df_filtered = df[df['cityname'] == "Charlotte"]  # filter by user-selected city (add input for city)
    
    user_data = {
        'bedrooms': user_bedrooms,
        'bathrooms': user_bathrooms,
        'square_feet': user_rent,  
        'pets?': 1 if user_pets == "Yes" else 0,
        'latitude':35.2016, #took random coordinates from charlotte apartment for placeholder
        'longitude':-80.8124 
    }
    
    user_df = pd.DataFrame([user_data])
    
    df_filtered = pd.concat([df_filtered, user_df], ignore_index=True)
    df_filtered, kmeans, scaler = cluster_data(df_filtered)
    
    user_cluster = kmeans.predict(scaler.transform(user_df[feature_cols]))
    similar_apartments = df_filtered[df_filtered['cluster'] == user_cluster[0]]
    
    model = train_regression_model(similar_apartments)
    predicted_rent = model.predict(user_df[feature_cols])
    
    st.write(f"Predicted Rent for your apartment: ${predicted_rent[0]:,.2f}") 
