import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as mat
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

st.set_page_config(page_title= "Classification of Glass Types", page_icon=":shopping_bag:", layout="wide")



# Load the Glass Identification dataset

url = "glass.data.csv"

column_names_all = { 
'Id': 'Id',
'RI':'Refractive Index',
'Na': 'Sodium', 
'Mg': 'Magnesium', 
'Al': 'Aluminum', 
'Si': 'Silicon', 
'K': 'Potassium', 
'Ca': 'Calcium', 
'Ba': 'Barium', 
'Fe': 'Iron',
'Type': 'Type'
}

column_names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
df = pd.read_csv(url, header=None, names=column_names)
df = df.drop(columns='Id')  # Drop the ID column as it's not useful for classification

X = df.drop(columns='Type')
y = df['Type']
#==============================


glass_types = {
	1: "Building Windows (float processed)",
	2: "Building Windows (non-float processed)",
	3: "Vehicle Windows (float processed)",
	4: "Vehicle Windows (non-float processed)",
	5: "Containers",
	6: "Tableware",
	7: "Headlamp"
}


#  PREDICTION PAGE



model2=pickle.load(open('glass.pkl','rb'))

st.title("Glass Type Prediction :wine_glass:")
st.subheader("Enter the features to identify the type of glass:")

# Define feature names for user input
feature_names = X.columns

#st.table(feature_names)
# Create input fields for each feature

# Create input fields for each feature with unique keys
inputs = {}
for i, feature in enumerate(feature_names):
    inputs[feature] = st.number_input("Enter value for "+ column_names_all[feature] + " (" + feature + ")", key=feature, value=0.0)

# Convert user inputs to a DataFrame
input_df = pd.DataFrame([inputs], columns=feature_names)

# Make predictions
if st.button('Classify'):
	prediction = model2.predict(input_df)
	#st.write(prediction)
	st.subheader("Predicted Glass Type is given below: ")
	pred_value = "Type " + str(prediction[0]) + ": " + glass_types[prediction[0]]
	st.subheader(pred_value)
	







