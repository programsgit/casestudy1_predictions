# The "Glass Identification" dataset is a popular dataset used for classification tasks involving different types of glass. It is often used to classify types of glass based on their chemical composition.
# Glass Identification Dataset Overview
# The Glass Identification dataset contains data about different types of glass, and the task is to classify the type of glass based on its chemical properties.
# Key Features:
# 1. RI: Refractive Index
# 2. Na: Sodium (in weight percentage)
# 3. Mg: Magnesium (in weight percentage)
# 4. Al: Aluminum (in weight percentage)
# 5. Si: Silicon (in weight percentage)
# 6. K: Potassium (in weight percentage)
# 7. Ca: Calcium (in weight percentage)
# 8. Ba: Barium (in weight percentage)
# 9. Fe: Iron (in weight percentage)
# 10. Type: Type of glass (target variable, with 7 classes representing different types of glass)
# Example of Classification for EDA
# Here’s a step-by-step guide to performing exploratory data analysis (EDA) and classification using the Glass Identification dataset:
# 1. Load the Dataset: The dataset is available from the UCI Machine Learning Repository.
# Summary
#========
# The Glass Identification dataset is used to classify different types of glass based on their chemical properties. 
# The Glass Identification shows how chemical properties can be used to differentiate between different types of glass.

# • EDA Tasks: Initial exploration of dataset structure, handling and visualizing features and their relationships, and understanding the distribution of classes.

# Select a classifier model
# Choosing the best classifier for this dataset depends on the characteristics of the data, such as the number of features, the number of classes, and the distribution of the data. Here are a few classifiers that are particularly well-suited for the Glass Identification dataset:
	# Random Forest:
	# ◦ Advantages:
		# ▪ Handles a large number of features effectively.
		# ▪ Provides feature importance, which can be useful for understanding which features are most influential.
		# ▪ Less prone to overfitting compared to a single decision tree.
	# ◦ Usage: Good choice for datasets with complex interactions between features.

	# Gradient Boosting Machines (like XGBoost) and Random Forest are highly recommended due to their ability to handle complex datasets with multiple features and classes.
	# • Support Vector Machines and k-Nearest Neighbors are also viable options but may require more tuning or may be less efficient with larger datasets.
	# • Logistic Regression can be used as a baseline or for comparison.


# • Classification Task: Train a model to classify different types of glass based on their chemical composition.


# • Model Evaluation: Use accuracy and classification reports to assess model performance.



# ===========
# ==========
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as mat
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



st.set_page_config(page_title= "Classification of Glass Types", page_icon=":wine_glass:", layout="wide")
st.title(":wine_glass: Glass Types Data Analysis ")


st.header("Glass Type Identification")


# Load Glass Identification dataset

url = "glass.data.csv"
column_names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']


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


df = pd.read_csv(url, header=None, names=column_names)
df = df.drop(columns='Id') 


# 2. Initial Exploration: Explore the dataset to understand its structure and check for basic statistics.

# ==========
# Display the first few rows
st.table(df.head())

# Summary statistics
st.header("Statistical Summary")

st.table(df.describe())



st.subheader("Glass Types ")
glass_types = {
	1: "Building Windows (float processed)",
	2: "Building Windows (non-float processed)",
	3: "Vehicle Windows (float processed)",
	4: "Vehicle Windows (non-float processed)",
	5: "Containers",
	6: "Tableware",
	7: "Headlamp"
}
st.table(glass_types)
 
st.subheader("Unique Glass Types:")
# unique_types = (df['Type'].sort_values().unique())
#st.write(unique_types)

st.table(df['Type'].value_counts().sort_index())



# Distribution of glass types








	





#===============

# Create a bar chart using Matplotlib to Show the distribution of glass types 
st.subheader("Distribution of different glass types")

type_counts = df['Type'].value_counts()
type_counts_sorted = type_counts.sort_index()
fig, ax = plt.subplots()
type_counts_sorted.plot(kind='bar', ax=ax)
ax.set_xlabel('Glass Type')  
ax.set_ylabel('Frequency')   
ax.set_title('Distribution of Glass Types')  
st.pyplot(fig)


# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)


#================

# MODEL TRAINING 

X = df.drop(columns='Type')
y = df['Type']
#==============================


# Histogram of Features
st.subheader("Histograms of Features")
# Create columns for layout
c1, c2 = st.columns(2)

# Loop through features and create histograms
for i, feature in enumerate(X.columns):
    fig, ax = plt.subplots()
    df[feature].hist(ax=ax, bins=30)
    ax.set_xlabel(column_names_all[feature])
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {feature}')  # Use f-string for dynamic title

    # Display histogram in the correct column
    if i % 2 == 0:
        c1.write(f'{column_names_all[feature]}')
        c1.pyplot(fig)
    else:
        c2.write(f'{column_names_all[feature]}')
        c2.pyplot(fig)

# Boxplot of Features by Type
st.subheader("Boxplots of Features by Glass Type")
# Create columns for layout
c3, c4 = st.columns(2)

# Loop through features and create boxplots
for i, feature in enumerate(X.columns):
    fig, ax = plt.subplots()
    sns.boxplot(x='Type', y=feature, data=df, ax=ax, palette='Set2')
    ax.set_xlabel('Glass Type')
    ax.set_ylabel(column_names_all[feature])
    ax.set_title(f'{column_names_all[feature]} ({feature}) value of different Glass Types')

    # Display boxplot in the correct column
    if i % 2 == 0:
        c3.write(f'{column_names_all[feature]}')
        c3.pyplot(fig)
    else:
        c4.write(f'{column_names_all[feature]}')
        c4.pyplot(fig)
#===============================
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=df['Type'].unique())

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=df['Type'].unique(), yticklabels=df['Type'].unique(), ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=45)

# Streamlit UI
st.title("Glass Classification App")

# Show confusion matrix
st.subheader("Confusion Matrix")
st.pyplot(fig)
# Display model accuracy
accuracy = accuracy_score(y_test, y_pred)
st.header("Model Accuracy: ")
accuracy2 = str(round((accuracy*100),2))
st.subheader(accuracy2+"%")

# Evaluate the model
st.header("Accuracy, Precision, Recall, and F1-score")
cr = classification_report(y_test, y_pred)
st.write(cr)








save_model = pickle.dump(model, open('glass.pkl', 'wb'))
