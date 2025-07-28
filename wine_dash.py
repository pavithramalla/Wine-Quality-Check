import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px 
import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

def load_model():
    with open("C:/Users/pavit/OneDrive/wine.sav", "rb") as file:
        return pickle.load(file)
model = load_model()

wine_data= pd.read_csv(r"C:\Users\pavit\OneDrive\winequality.csv")
print(wine_data['quality'].value_counts())
print("Model and data loaded successfully.") 

st.title("Wine Quality Check Dashboard")
st.set_page_config(page_title="Wine Quality Check", layout="wide", initial_sidebar_state="auto")

st.subheader("Checking Board")

with st.form(key='input_form'):
    st.subheader("Enter Wine Features Below")
    col1, col2, col3 = st.columns(3)
    with col1:
         fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.0)
         volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.7)
         citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.0)
         residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=1.9)
    with col2:     
         chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.076)
         free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0, max_value=100, value=11)
         total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0, max_value=300, value=34)
         density = st.number_input("Density", min_value=0.0, max_value=2.0, value=0.9978)
    with col3:    
         pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=3.51)
         sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.56)
         alcohol = st.number_input("Alcohol", min_value=0.0, max_value=20.0, value=9.4)
         wine_type = st.selectbox("Wine Type", options=["red", "white"])
         wine_type_numeric = 0 if wine_type == "red" else 1

    submit_button = st.form_submit_button(label='Predict Quality')

if submit_button:
    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol],
        'type': [wine_type_numeric]
    })

    prediction = model.predict(input_data)
    predicted_value = prediction[0]
    st.write("Raw Prediction Value:", predicted_value)


    if predicted_value == 0:
       st.error("Predicted Wine Quality: Bad Quality")
       quality_label = "Bad"
    elif predicted_value == 1:
       st.warning("Predicted Wine Quality: Neutral Quality")
       quality_label = "Neutral"
    elif predicted_value == 2:
       st.success("Predicted Wine Quality: Good Quality")
       quality_label = "Good"
    else:
       st.info(f"Unknown Quality: {predicted_value}")


    input_data['quality'] = predicted_value
    wine_data = pd.concat([wine_data, input_data], ignore_index=True)

fig = make_subplots(rows=1, cols=3, subplot_titles=( "type vs pH", "Alcohol vs Quality","Wine Quality"),horizontal_spacing=0.15)
fig.add_trace(go.Scatter(x=wine_data['type'], y=wine_data['pH'], mode='markers',marker=dict(color='orange'),name='type vs pH'), row=1, col=1)
fig.add_trace(go.Box(x=wine_data['quality'], y=wine_data['alcohol'], marker_color='darkgreen', name='Alcohol % by Wine Quality'), row=1, col=2)
fig.add_trace(go.Histogram(x=wine_data['quality'], marker_color='royalblue', name='Wine Quality'), row=1, col=3)

fig.update_layout(height=300,width=900,paper_bgcolor='white',plot_bgcolor='white',font=dict(color="black"),
                  xaxis1=dict(tickfont=dict(size=14, color='black')),yaxis1=dict(tickfont=dict(size=14, color='black')),
                  xaxis2=dict(tickfont=dict(size=14, color='black')),yaxis2=dict(tickfont=dict(size=14, color='black')),
                  xaxis3=dict(tickfont=dict(size=14, color='black')),yaxis3=dict(tickfont=dict(size=14, color='black')),
                  title="Graphical Information", showlegend=True,margin=dict(l=50, r=50,t=60, b=40))
fig.show()
st.plotly_chart(fig, use_container_width=True)
