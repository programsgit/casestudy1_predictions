# ============ DATA ANALYTICS ==============
# ============== CASE STUDY ================

import streamlit as st

pg = st.navigation([
st.Page("menu.py",title="Home Page"),
#st.Page("dashboard_netflix.py",title="Netflix Trends Dashboard"),

st.Page("glass.py",title="Glass Type - Classification"),
st.Page("glass_pred.py",title="Glass Type - Prediction "),

st.Page("car_price_eda.py",title="Car Price - EDA"),
st.Page("car_price_prediction.py",title="Car Price - Prediction"),

st.Page("clustering.py",title="Mall Customers - Clustering"),
st.Page("clustering_prediction.py",title="Mall Customers - Prediction"),

])

pg.run()
