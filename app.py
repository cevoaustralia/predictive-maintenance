import streamlit as st
import warnings
import pandas as pd
import numpy as np
import os
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Importing Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error    
sns.set(color_codes=True)

# Define the main function to run the app
def main():
    st.title("Logistic Regression App")
    st.sidebar.title("Upload Dataset")

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("## Dataset")
        st.write(df)

        st.write("## Dataset Statistics")
        st.write(df.describe())

        # Display the multiple selection box
        target_variable = st.sidebar.selectbox("Select Target Variable", df.columns)
        st.write(target_variable)

        # Dropping unnecessary columns
        st.write("## Dropping unnecessary columns and standardizing the variables")
        df = df.drop(['UDI', 'Product ID','Type'], axis=1)
        scaler=StandardScaler()
        df = scaler.fit_transform(df)
        df = pd.DataFrame(df)
        st.write("## Standardized variables")
        st.write(df.head(5))
        X = df[[0,1,2,3,4]].copy()
        Y = df[[5]].copy()

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        # Creating a Linear Regression model
        model = LinearRegression()
        # Training the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)


        st.write("Mean Squared Error:", mse)



# Run the main function to start the app
if __name__ == "__main__":
    main()
