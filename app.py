import pandas as pd 
import numpy as np 
import pickle as pi
import streamlit as st


model = pi.load(open("model.pkl","rb"))

st.header("CAR PRICE PREDICTION MACHINE LEARNING MODEL ")

cars_data = pd.read_csv("Cardetails.csv")

def get_brand_name(car_name): 
    car_name = car_name.split(" ")[0]
    return car_name.strip()
cars_data["name"] = cars_data["name"].apply(get_brand_name)

name =     st.selectbox("SELECT CAR BRAND", cars_data["name"].unique())
fuel =     st.selectbox("FUEL TYPE", cars_data["fuel"].unique())
seller_type =   st.selectbox("SELLER TYPE", cars_data["seller_type"].unique())
transmission =    st.selectbox("TRANSMISSION TYPE ", cars_data["transmission"].unique())
owner = st.selectbox("SELLEAR TYPE",cars_data["owner"].unique())
mileage =   st.slider("CAR MILEAGE",10,40)
engine =   st.slider("ENGINE CAPACITY",700,5000)
max_power =    st.slider("MAX POWER ",0,250)
year =     st.slider("CAR MANUFACTURED YEAR",1994,2024)
seats =     st.slider("CAR OF SEAT",5,10)
km_driven = st.slider("DISTANCE TRAVEL",11,200000)

if st.button("Predict"):
    input_data_model = pd.DataFrame(
     [[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]], 
     columns=["name","year","km_driven","fuel","seller_type","transmission","owner","mileage","engine","max_power","seats"])
    
    input_data_model["owner"].replace(["First Owner", "Second Owner", "Third Owner",
       "Fourth & Above Owner", "Test Drive Car"], 
                          [1,2,3,4,5],inplace=True)
    input_data_model["fuel"].replace(["Diesel", "Petrol", "LPG", "CNG"],[1,2,3,4],inplace=True)
    input_data_model["seller_type"].replace(["Individual", "Dealer", "Trustmark Dealer"],[1,2,3],inplace=True)
    input_data_model["transmission"].replace(["Manual", "Automatic"],[1,2],inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                          [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
                          ,inplace=True)

    car_price = model.predict(input_data_model)
    st.markdown("CAR PRICE VALUE :-" + str (car_price[0]))


