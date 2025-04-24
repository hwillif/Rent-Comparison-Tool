import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import euclidean

# Import Cleaned Data
apartments = pd.read_csv('clean_apartments.csv')
apartments['cityname'] = apartments["cityname"].astype(str)


# Page Title
st.title('Rent Comparison Tool')
st.markdown("Hayden Williford, Matthew Rostar, Jack Callahan, Lillian Bowling")

st.header("Your Apartment Details:")

# Create Fake Columns to get city name
city_names = apartments['cityname']

# Create Global Variables
user_cluster = 0


col12, col22, col32 = st.columns(3)
with col12:
    # Get User City Name
    user_city = st.text_input("City Name?")

    if user_city:
        if user_city in city_names.values:
            st.write(f"City Selected: {user_city}")
        else:
            st.error(f"{user_city} is not an approved city")
            st.error("Please try another city")


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

    # User Square Footage of Apartment Text Box
    user_sqft = st.text_input("Square Footage?")

    # Display the Square Footage Value and Restrict it to numbers
    if user_sqft:
        if user_sqft.isdigit():
            number = int(user_sqft)
            st.write(f"Square Footage: {user_sqft} ft²")
            user_sqft = float(user_sqft)
        else:
            st.error("Please enter a valid whole number") 


with col2:
    # User Bedrooms Text Box
    user_bedrooms = st.text_input("Number of Bedrooms?")

    # Display the Bedrooms value and restrict text to numbers
    if user_bedrooms:
        try:
            value = float(user_bedrooms)
            if 0 <= value <= 4 and (value * 2).is_integer():
                st.write(f"Number of Bedrooms: {value}")
                user_bedrooms = float(user_bedrooms)
            else:
                st.error("Number must be between 0 and 4")
        except ValueError:
            st.error("Please enter a valid number.")

    #Pets Allowed Drop Down
    user_pets = st.selectbox("Pets Allowed?", 
                             ("No","Yes"), 
                             index = None)
    if user_pets != None:
        st.write(f"Pets Allowed: {user_pets}")


with col3:
    # User Bathrooms Text Box
    user_bathrooms = st.text_input("Number of Bathrooms?")

    # Display the Number of Bathrooms value and restrict values
    if user_bathrooms :
        try:
            value = float(user_bathrooms)
            if 0 <= value <= 4 and (value * 2).is_integer():
                st.write(f"Number of Bathrooms: {value}")
                user_bathrooms = float(user_bathrooms)
            else:
                st.error("Number must be between 0 and 10, in 0.5 steps.")
        except ValueError:
            st.error("Please enter a valid number.")
        
# Feature Names of Training Data
features = ['title', 'cityname', 'price', 'bedrooms', 'bathrooms', 'square_feet', 'pets?','longitude', 'latitude']

# Create User Dataframe
user = pd.DataFrame({
    "title": ['Users Apartment'],
    "cityname": [user_city],
    "price": [user_rent],
    "bedrooms": [user_bedrooms],
    "bathrooms": [user_bathrooms],
    "square_feet": [user_sqft],
    "pets?": [user_pets],
    "longitude": [None],
    "latitude": [None]
})


# Correct User_Pets from Yes and No to 1 and 0
user['pets?'] = user['pets?'].replace({'Yes': 1, 'No': 0})


# Make Training Dataset
filtered_apartments = apartments[features]
train_df = pd.concat([user, filtered_apartments]).reset_index(drop=True)

# USER INDEX IS 0 (FIRST ROW IN TRAIN DF)
record_index = 0

# Create Dataframe for Map
map_df = train_df[train_df['cityname'] == user_city].reset_index(drop=True)
map_df = map_df.drop(columns = ['cityname', 'price', 'bedrooms', 'bathrooms', 'square_feet', 'pets?'], axis=1)
st.write("Dataset of The Map Data")
st.write(map_df)


# Filter train_df to only the city the user entered
train_df = train_df[train_df['cityname'] == user_city].reset_index(drop=True)
train_df = train_df.drop(columns = ['cityname', 'price','longitude','latitude'], axis=1)

# Display Dataset Before Kmeans
st.write('Dataset of User Entered Data and Apartment Dataset')
st.write(train_df.head())

kmeans_features = ['bedrooms', 'bathrooms', 'square_feet', 'pets?']


# Create Function to do Kmeans on Filtered dataframe
def run_kmeans(df, columns, n_clusters = 5):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns])

    kmeans = KMeans(n_clusters= n_clusters, random_state = 0)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Get User Cluster
    user_cluster = df.iloc[0, 4]
    user_point = df.loc[0, columns].values

    # Filter Dataframe to only those in the users cluster
    user_cluster_df = df[df['cluster'] == user_cluster].copy()

    # Calculate Distance from user record
    user_cluster_df['distance_from_user'] = user_cluster_df[columns].apply(lambda row: euclidean(row.values, user_point), axis=1)

    return user_cluster_df



# Create a Button to do K Means Clustering
if st.button("Find Similar Apartments"):
    kmeans_results = run_kmeans(train_df, columns= kmeans_features, n_clusters= 5)
    st.write('Dataframe after Clustering')
    st.write(kmeans_results)

    st.header("Top 5 Recommendations")
    st.write(kmeans_results.sort_values(by='distance_from_user').head(5))

