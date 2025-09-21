import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor


st.markdown("<h1 style='text-align: center;'>🏡 Land Price Predictor 🏡</h1>", unsafe_allow_html=True)

st.subheader(" For selected indian cities 💰")
st.subheader("made By Priyum Deb")
data=pd.read_csv("city_price.csv")

X=data[["city","sqft"]]
Y=data["price"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

preprocessor= ColumnTransformer([
    ("S_sqft",StandardScaler(),["sqft"]),
    ("S_city",OneHotEncoder(),["city"])
])

X_train=preprocessor.fit_transform(X_train)
X_test=preprocessor.transform(X_test)

model=RandomForestRegressor()
model.fit(X_train,Y_train)

city=st.selectbox("choose city",['Siliguri','Kolkata','Mumbai','Delhi','Bengaluru','Chennai','Hyderabad'])
ft=st.number_input('enter sqfeet', min_value=100,value=100)

a=pd.DataFrame([{"city":city,"sqft":ft}])
a_ready=preprocessor.transform(a)

if st.button("predict"):
    Predict=model.predict(a_ready)[0]
    print("₹",Predict)
    st.success(f"Predicted Price : ₹{Predict}")
st.caption("⚠️This is a demo model trained on limited data" \
"Predictions may not reflect actual market prices⚠️")