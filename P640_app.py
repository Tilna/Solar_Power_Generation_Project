#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import joblib


# In[2]:


# Load trained model
model = joblib.load("xgboost_solar_model.pkl")


# In[3]:


st.set_page_config(page_title="Solar Power Predictor", layout="centered")
st.title("☀ Solar Power Generation Prediction")
st.write("Enter environmental parameters to predict solar power generation.")


# In[4]:


## input
distance = st.number_input("Distance to Solar Noon (radians)", value=0.5)
temperature = st.number_input("Temperature (°C)", value=25.0)
wind_direction = st.number_input("Wind Direction (0-360 degrees)", value=180.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=5.0)
sky_cover = st.slider("Sky Cover (0 = clear, 4 = fully covered)", 0, 4, 1)
humidity = st.number_input("Humidity (%)", value=60.0)
avg_wind = st.number_input("Average Wind Speed (3-hour) (m/s)", value=5.0)
pressure = st.number_input("Average Pressure (inches Hg)", value=30.0)


# In[6]:


if st.button("Predict Power Generation"):

    wind_dir_sin = np.sin(np.deg2rad(wind_direction))
    wind_dir_cos = np.cos(np.deg2rad(wind_direction))

    input_data = np.array([
        distance,
        temperature,
        wind_speed,
        sky_cover,
        humidity,
        avg_wind,
        pressure,
        wind_dir_sin,
        wind_dir_cos
    ]).reshape(1, -1)

    prediction = model.predict(input_data)

    st.success(f"🔋 Predicted Power Generated: {prediction[0]:.2f} Joules")


# In[ ]:




