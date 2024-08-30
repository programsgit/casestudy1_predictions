# 2. Make a web app for implementing clustering on mall dataset.

# Prediction Page
import streamlit as st
import pickle

st.set_page_config(page_title= "Clustering of Mall Customers", page_icon=":shopping_bag:", layout="wide")
st.title(":people_holding_hands::shopping_trolley: Mall Customers - Prediction :shopping_bags:")

#CustomerID	Gender	Age	Annual Income (k$)	Spending Score (1-100)
model1=pickle.load(open('mall_kmeans.pkl','rb'))

st.header("Prediction :gift:")
c1, c2 = st.columns(2)


gender_list = {0:"Female", 1: "Male"}
selected_gender = c1.selectbox("Enter Gender", list(gender_list.values()))
n1 = [key for key, value in gender_list.items() if value == selected_gender][0]
# c1.write(f"Selected Gender: {selected_gender}")
# c1.write(f"Value: {n1}")


n2=int(c1.number_input("Enter Age", step=1, min_value=18, max_value=120))
n3=int(c1.number_input("Enter Annual Income (k$)", step=1, min_value=0, max_value=1000000))



sample=[[n1,n2,n3]]

if(st.button("Predict the category of the Customer")):
    t = model1.predict(sample)
    st.write("Customer in category :")
    #st.write(t)
    if (t == 0):
        st.write("Customer with Low Shopping score")
    elif (t == 1):
        st.write("Customer with Normal Shopping score")
    elif (t == 2):
        st.write("Customer with High Shopping score")
    elif (t == 3):
        st.write("Customer with Very High Shopping score")   
    elif (t == 4):
        st.write("Customer with Low Shopping score")    		
    else:
        st.write("Shopping score not listed")


