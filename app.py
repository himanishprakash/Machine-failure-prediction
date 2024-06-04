import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title of the web app
st.title('Machine Failure Prediction')

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    type_ = st.sidebar.selectbox('Type', ('L', 'M', 'H'))  # Updated categories to 'L', 'M', 'H'
    air_temp = st.sidebar.slider('Air temperature [K]', 290, 310, 298)  # Adjust the range according to your data
    process_temp = st.sidebar.slider('Process temperature [K]', 0, 400, 300)  # Adjust the range according to your data
    rotational_speed = st.sidebar.slider('Rotational speed [rpm]', 0, 3000, 1500)  # Adjust the range according to your data
    torque = st.sidebar.slider('Torque [Nm]', 0, 100, 50)  # Adjust the range according to your data
    tool_wear = st.sidebar.slider('Tool wear [min]', 0, 300, 150)  # Adjust the range according to your data
    data = {
        'Type': type_,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Load and preprocess data
data = pd.read_csv('predictive_maintenance (1).csv')

# Drop irrelevant columns
data = data.drop(["UDI", "Product ID"], axis=1)

# Convert categorical column 'Type' to numerical codes
data['Type'] = data['Type'].astype('category').cat.codes

# Fill missing values in numeric columns with the mean
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Splitting data
X = data.drop(['Failure Type', 'Target'], axis=1)
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure that the user input is in the same format as the training data
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})  # Ensure mapping matches the training data encoding

# Building the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(df)

# Display prediction result
st.subheader('Prediction')
st.write('Failure' if prediction[0] == 1 else 'No Failure')

# Model accuracy
st.subheader('Model Accuracy')
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f'Accuracy: {accuracy:.2f}')
