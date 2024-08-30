# 2. Make a web app for car prediction using Linear algorithm(Multi vari-
# able), do eda in one page and prediction in another page.

# PREDICTION page

import streamlit as st
import pickle
import locale
import joblib

locale.setlocale(locale.LC_ALL, '' )

#lmodel1 = pickle.load(open('car_price1.pkl', 'rb'))
#lmodel2 = pickle.load(open('car_price2.pkl', 'rb'))
#lmodel3 = pickle.load(open('car_price3.pkl', 'rb'))
#lmodel4 = pickle.load(open('car_price4.pkl', 'rb'))


lmodel1 = joblib.load('car_price1.pkl')
lmodel2 = joblib.load('car_price2.pkl')
lmodel3 = joblib.load('car_price3.pkl')
lmodel4 = joblib.load('car_price4.pkl')


st.header("Prediction")

# Name	Type	Seats	CC	Min Price (Lakh)	Max Price (Lakh)	Range (kmpl)	Price

n1=int(st.number_input("Name", step=1, min_value=0, max_value=12))
n2=int(st.number_input("Type (0:petrol, 1: Diesel)", step=1, min_value=0, max_value=1))

#st.radio("Type", ["Petrol","Diesel"])


n3=int(st.number_input("Seats", step=1, min_value=4, max_value=9))
n4=int(st.number_input("CC", step=100, min_value=800, max_value=3000))
n5=int(st.number_input("Min Price (Lakh)", step=1, min_value=2, max_value=200))
n6=int(st.number_input("Max Price (Lakh)", step=1, min_value=10, max_value=200))
n7=int(st.number_input("Range (kmpl)", step=1, min_value=5, max_value=50))

sample1=[[n1,n2,n3,n4, n5,n6,n7]]

def priceformat(x):
    n=0
    try:
        n = round(x.flatten()[0],2)
        #n = locale.currency(n, grouping=True )
    except:
        n=0
    return n
if st.button("Predict the price"):
    t1=lmodel1.predict(sample1)
    t2=lmodel2.predict(sample1)
    t3=lmodel3.predict(sample1)
    t4=lmodel4.predict(sample1)

    if (t1):
        st.header("Predicted price of the car is:")
        st.subheader("Predicted price by 4 methods:")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.subheader("Linear Regression")
            st.subheader(priceformat(t1))
        with c2:
            st.subheader("Ridge Model")
            st.subheader(priceformat(t2))
        with c3:
            st.subheader("Lasso Model")
            st.subheader(priceformat(t3))
        with c4:
            st.subheader("ElasticNet Model")
            st.subheader(priceformat(t4))
   

    else:
        st.write("Not listed")
