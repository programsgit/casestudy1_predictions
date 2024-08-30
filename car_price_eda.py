# 2. Make a web app for car prediction using Linear algorithm(Multi vari-
# able), do eda in one page and prediction in another page.
# car dataset to analyze the price range, Range, CC, variants, and their on-road prices.
# It is made only for beginners who want to do simple analysis.

# Columns Description:

    # Name: Shows the name of Car
    # Min Price (in Lakh): Shows the minimum price of the car model.
    # Max Price (In lakh): Shows the max price of the model of the car can be.
    # Range (KMPL): This shows the milage of the car.
    # CC: The CC no. of the car.
    # Seats: Shows the seats available in the car.
    # Varient: This tells about the variant available in the market.
    # Type: Tells more about the type of the car (Petrol/diesel/EV).
    # On-Road Price: This shows the total price of the car to buy in the city Delhi.

# EDA page


import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics as mat
import seaborn as sns
import joblib
st.set_page_config(page_title="Car Price",page_icon=":car:",layout="wide")

st.title(":car: Car Price Data Analysis :car: ")
df = pd.read_csv("car_price.csv")
st.subheader("Car Price Dataset")
st.dataframe(df.head())
#================

# df.drop(columns=['Name'], inplace=True)
# df.drop(columns=['Type'], inplace=True)
st.header("Correlation matrix of features")

le = LabelEncoder()
df['Name'] = le.fit_transform(df['Name'])
df['Type'] = le.fit_transform(df['Type'])


corr = df.corr()
fig2 = px.imshow(corr, text_auto =True)
st.plotly_chart(fig2, use_container_width = True)

#=== get variables
x = df.drop(columns=['Price'])
y = df[['Price']]

c1, c2 = st.columns(2)
c1.subheader("Features are")
c1.dataframe(x)
c2.subheader("Labels are")
c2.dataframe(y)

#=== get linear Regression model and train it
lmodel = LinearRegression()
lmodel.fit(x, y)





#=======================


st.subheader("Column heads")
st.dataframe(df.columns)


# split data as two tables - null values as testing table and not null as training table
training_data = df[df['Price'].isnull() == False ]
testing_data = df[df['Price'].isnull() == False ]

#==============
c1, c2 = st.columns(2)
c1.subheader("Shape of training data")
c1.write(training_data.shape)

c2.subheader("Shape of testing data")
c2.write(testing_data.shape)
#==================
c1.subheader("Null values in training data")
c1.write(training_data.isnull().sum())

c2.subheader("Null values in testing data")
c2.write(testing_data.isnull().sum())
#==================

c1.subheader("Training data")
c1.table(training_data.head())

c2.subheader("Testing data")
c2.table(testing_data.head())
#==================


xtrain = training_data.drop("Price", axis=1)
ytrain = training_data['Price']

xtest = testing_data.drop("Price", axis=1)
ytest = testing_data['Price']


#==============
col1, col2, col3, col4  = st.columns(4)
col1.subheader("Features of Training data")
col1.table(xtrain.head())

col2.subheader("Labels of Training data")
col2.table(ytrain.head())

col3.subheader("Features of Testing data")
col3.table(xtest.head())

col4.subheader("Labels of Testing data")
col4.table(ytest.head())
#==================

# ===== Get the Models ====
rid  = Ridge()
las  = Lasso()
enet = ElasticNet()

# ===== Training the Models with our training data ====
rid.fit(xtrain, ytrain)
las.fit(xtrain, ytrain)
enet.fit(xtrain, ytrain)

# ===== Storing these trained Models to a file
# saved_model = pickle.dump(lmodel, open('car_price1.pkl', 'wb'))
# m1 = pickle.dump(rid, open('car_price2.pkl','wb'))
# m2 = pickle.dump(las, open('car_price3.pkl','wb'))
# m3 = pickle.dump(enet, open('car_price4.pkl','wb'))

m1 = joblib.dump(lmodel, 'car_price1.pkl')
m2 = joblib.dump(rid, 'car_price2.pkl')
m3 = joblib.dump(las, 'car_price3.pkl')
m4 = joblib.dump(enet,'car_price4.pkl')

# ===== Predicting values for our test data ==============
ypred1 = rid.predict(xtest)
ypred2 = las.predict(xtest)
ypred3 = enet.predict(xtest)


st.header("Comparison of different Models")
mse1 = mat.mean_squared_error(ypred1,ypred2)
mse2 = mat.mean_squared_error(ypred2,ypred3)
mse3 = mat.mean_squared_error(ypred1,ypred3)

r2_1 = mat.r2_score(ypred1,ypred2)
r2_2 = mat.r2_score(ypred2,ypred3)
r2_3 = mat.r2_score(ypred1,ypred3)

mae1 = mat.mean_absolute_error(ypred1,ypred2)
mae2 = mat.mean_absolute_error(ypred2,ypred3)
mae3 = mat.mean_absolute_error(ypred1,ypred3)

c2a, c3,c4,c5 = st.columns(4)

c2a.subheader("Model  Name")
c2a.write("Ridge")
c2a.write("Lasso")
c2a.write("ElasticNet")

c3.subheader("Mean Squared Error (MSE)")
c3.write(mse1)
c3.write(mse2)
c3.write(mse3)

c4.subheader("R2 Score  (Score)")
c4.write(r2_1)
c4.write(r2_2)
c4.write(r2_3)

c5.subheader("Mean Absolute Error (MAE)")
c5.write(mae1)
c5.write(mae2)
c5.write(mae3)
st.divider()



st.header("Price Prediction of different Models")

testing_data['Ridge_price']=ypred1
testing_data['Lasso_price']=ypred2
testing_data['Enet_price'] =ypred3

st.dataframe(testing_data)

fig,ax=plt.subplots(figsize=(2,2))
ax.plot(ypred1,c='g',marker='+')
ax.plot(ypred2,c='b',marker='*')
ax.plot(ypred3,c='r',marker='.')

st.pyplot(fig)


#==== Scatter plot
st.header("Price Prediction of different Models")


testing_data['id'] = range(1, 1+len(testing_data))

testing_data['Ridge_price']=ypred1
testing_data['Lasso_price']=ypred2
testing_data['Enet_price'] =ypred3

st.dataframe(testing_data)


#==== Scatter plot
st.header("price Prediction of different Models")

col5, col6, col7 = st.columns(3)
 
fig1 = px.scatter(testing_data, x=testing_data['id'], y=testing_data['Ridge_price'])
col5.plotly_chart(fig1)

fig2 = px.scatter(testing_data, x=testing_data['id'], y=testing_data['Lasso_price'])
col6.plotly_chart(fig2)

fig3 = px.scatter(testing_data, x=testing_data['id'], y=testing_data['Enet_price'])
col7.plotly_chart(fig3)





