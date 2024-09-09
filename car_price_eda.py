# 2. Make a web app for car prediction using Linear algorithm(Multi variable), do eda in one page and prediction in another page.
# Car dataset to analyze the price range, Range, CC, variants, and their on-road prices.

# Columns Description:

# Name: Shows the name of Car
# Min Price (in Lakh): Shows the minimum price of the car model.
# Max Price (In lakh): Shows the max price of the model of the car can be.
# Range (KMPL): This shows the mileage of the car.
# CC: The CC no. of the car.
# Seats: Shows the seats available in the car.
# Variant: This tells about the variant available in the market.
# Type: Tells more about the type of the car (Petrol/diesel/EV).
# On-Road Price: This shows the total price of the car to buy in the city Delhi.

# EDA page

import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics as mat
import matplotlib.pyplot as plt

st.set_page_config(page_title="Car Price", page_icon=":car:", layout="wide")

st.title(":car: Car Price Data Analysis :car: ")
df = pd.read_csv("car_price.csv")
st.subheader("Car Price Dataset")
st.dataframe(df.head())

# Correlation matrix of features
st.header("Correlation matrix of features")
le = LabelEncoder()
df['Name'] = le.fit_transform(df['Name'])
df['Type'] = le.fit_transform(df['Type'])
corr = df.corr()
fig2 = px.imshow(corr, text_auto=True)
st.plotly_chart(fig2, use_container_width=True)

# Get variables
x = df.drop(columns=['Price'])
y = df[['Price']]

c1, c2 = st.columns(2)
c1.subheader("Features are")
c1.dataframe(x)
c2.subheader("Labels are")
c2.dataframe(y)

# Split data as two tables - null values as testing table and not null as training table
training_data = df[df['Price'].isnull() == False]
testing_data = df[df['Price'].isnull() == False]

c1, c2 = st.columns(2)
c1.subheader("Shape of training data")
c1.write(training_data.shape)

c2.subheader("Shape of testing data")
c2.write(testing_data.shape)

c1.subheader("Null values in training data")
c1.write(training_data.isnull().sum())

c2.subheader("Null values in testing data")
c2.write(testing_data.isnull().sum())

c1.subheader("Training data")
c1.table(training_data.head())

c2.subheader("Testing data")
c2.table(testing_data.head())

xtrain = training_data.drop("Price", axis=1)
ytrain = training_data['Price']

xtest = testing_data.drop("Price", axis=1)
ytest = testing_data['Price']

col1, col2, col3, col4 = st.columns(4)
col1.subheader("Features of Training data")
col1.table(xtrain.head())

col2.subheader("Labels of Training data")
col2.table(ytrain.head())

col3.subheader("Features of Testing data")
col3.table(xtest.head())

col4.subheader("Labels of Testing data")
col4.table(ytest.head())

# Get the Models
lmodel = LinearRegression()
rid = Ridge()
las = Lasso()
enet = ElasticNet()

# Training the Models with our training data
lmodel.fit(xtrain, ytrain)
rid.fit(xtrain, ytrain)
las.fit(xtrain, ytrain)
enet.fit(xtrain, ytrain)

# Storing these trained Models to a file
pickle.dump(lmodel, open('car_price1.pkl', 'wb'))
pickle.dump(rid, open('car_price2.pkl', 'wb'))
pickle.dump(las, open('car_price3.pkl', 'wb'))
pickle.dump(enet, open('car_price4.pkl', 'wb'))

# Predicting values for our test data
ypred0 = lmodel.predict(xtest).flatten()
ypred1 = rid.predict(xtest)
ypred2 = las.predict(xtest)
ypred3 = enet.predict(xtest)


st.header("Price Prediction of different Models")
testing_data['id'] = range(1, 1+len(testing_data))
testing_data['Linear_price'] = ypred0
testing_data['Ridge_price'] = ypred1
testing_data['Lasso_price'] = ypred2
testing_data['Enet_price'] = ypred3

st.dataframe(testing_data)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ypred0, c='r', marker='*', label='Linear')
ax.plot(ypred1, c='g', marker='+', label='Ridge')
ax.plot(ypred2, c='b', marker='*', label='Lasso')
ax.plot(ypred3, c='r', marker='.', label='ElasticNet')
ax.legend()

st.pyplot(fig)

st.header("Price Prediction of different Models")

col5, col6 = st.columns(2)

fig1 = px.scatter(testing_data, x='id', y='Linear_price', title='Linear Price Predictions')
col5.plotly_chart(fig1)

fig1 = px.scatter(testing_data, x='id', y='Ridge_price', title='Ridge Price Predictions')
col6.plotly_chart(fig1)

fig2 = px.scatter(testing_data, x='id', y='Lasso_price', title='Lasso Price Predictions')
col5.plotly_chart(fig2)

fig3 = px.scatter(testing_data, x='id', y='Enet_price', title='ElasticNet Price Predictions')
col6.plotly_chart(fig3)

# Metrics
r2 = mat.r2_score(ytest, ypred0)
mae = mat.mean_absolute_error(ytest, ypred0)
mse = mat.mean_squared_error(ytest, ypred0)
rmse = mse ** 0.5


# Calculate residuals for each model
residuals_linear = ytest - ypred0
residuals_ridge = ytest - ypred1
residuals_lasso = ytest - ypred2
residuals_enet = ytest - ypred3

# Function to plot residuals
def plot_residuals(residuals, title):
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted size for better fit in columns
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    return fig

# Streamlit layout
st.header("Residual Plots for Different Models")

# Create columns for the plots
col1, col2, col3, col4 = st.columns(4)

# Linear Regression residuals plot
with col1:
    st.subheader("Linear Regression")
    fig_linear = plot_residuals(residuals_linear, 'Residuals Distribution for Linear Regression')
    st.pyplot(fig_linear)

# Ridge Regression residuals plot
with col2:
    st.subheader("Ridge Regression")
    fig_ridge = plot_residuals(residuals_ridge, 'Residuals Distribution for Ridge Regression')
    st.pyplot(fig_ridge)

# Lasso Regression residuals plot
with col3:
    st.subheader("Lasso Regression")
    fig_lasso = plot_residuals(residuals_lasso, 'Residuals Distribution for Lasso Regression')
    st.pyplot(fig_lasso)

# ElasticNet Regression residuals plot
with col4:
    st.subheader("ElasticNet Regression")
    fig_enet = plot_residuals(residuals_enet, 'Residuals Distribution for ElasticNet Regression')
    st.pyplot(fig_enet)
	
	
# Display results
st.header("Comparison of different Models")
st.subheader('Model Performance Metrics')
mse0 = mat.mean_squared_error(ytest, ypred0)
mse1 = mat.mean_squared_error(ytest, ypred1)
mse2 = mat.mean_squared_error(ytest, ypred2)
mse3 = mat.mean_squared_error(ytest, ypred3)

rmse0 = mse0 ** 0.5
rmse1 = mse1 ** 0.5
rmse2 = mse2 ** 0.5
rmse3 = mse3 ** 0.5

r2_0 = mat.r2_score(ytest, ypred0)
r2_1 = mat.r2_score(ytest, ypred1)
r2_2 = mat.r2_score(ytest, ypred2)
r2_3 = mat.r2_score(ytest, ypred3)

mae0 = mat.mean_absolute_error(ytest, ypred0)
mae1 = mat.mean_absolute_error(ytest, ypred1)
mae2 = mat.mean_absolute_error(ytest, ypred2)
mae3 = mat.mean_absolute_error(ytest, ypred3)

col_1, col_2, col_3, col_4, col_5 = st.columns(5)

with col_1:
    st.header("Model Name")
    st.subheader("Linear")
    st.subheader("Ridge")
    st.subheader("Lasso")
    st.subheader("ElasticNet")

with col_2:
    st.header("Mean Squared Error (MSE)")
    st.subheader(mse0)
    st.subheader(mse1)
    st.subheader(mse2)
    st.subheader(mse3)

with col_3:
    st.header("R2 Score (Score)")
    st.subheader(r2_0)
    st.subheader(r2_1)
    st.subheader(r2_2)
    st.subheader(r2_3)

with col_4:
    st.header("Mean Absolute Error (MAE)")
    st.subheader(mae0)
    st.subheader(mae1)
    st.subheader(mae2)
    st.subheader(mae3)

with col_5:
    st.header("Root Mean Squared Error (RMSE)")
    st.subheader(rmse0)
    st.subheader(rmse1)
    st.subheader(rmse2)
    st.subheader(rmse3)
st.divider()
