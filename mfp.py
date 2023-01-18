import streamlit as st 
import numpy as np
import plotly.express as px 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 

# Make container 
header = st.container()
data_sets = st.container()
model_training = st.container()

with header: 
    st.title("Final Year Project App")
    st.text("We will work with parameter dataset of Induction Motor collected through sensors")
    
with data_sets:
    st.header("Parameter Dataset")
  
    #import data 
    df = pd.read_csv("FYP_Dataset.csv")
    df = df.dropna()
    st.write(df.head(10))
    
    fig = px.scatter(df, x="TESTING AMP", y="VOLTS ", hover_name="Fault ", color="Fault ",
     width=None, height=None)
    st.write(fig)
     
    year_option = df['Year '].unique().tolist()
    Years = st.sidebar.selectbox("You can see the Fault conditions with rpm per Month ", year_option,0)
    
    fig1 = px.box(df , x="Fault ", y="VOLTS ", color="Fault ", hover_name="Month ",points= 'all', animation_group="Fault ",
                     animation_frame='Year ')
    st.write(fig1)
    
    fig2 = px.scatter(df , x="Fault ", y="TEMPERATURE", color="Fault ", hover_name="Month ",animation_group="Fault ")                
    st.write(fig2)
    
    fig3 = px.scatter(df , x="Fault ", y="HUMIDITY ", color="Fault ", hover_name="Month ",animation_group="Fault ")
    st.write(fig3)
    
    
with model_training:
    
    st.header("Machine Learning Algorithm Results")
    df = df.drop(778)
    features , lables = st.columns(2)
    with features:
        st.text("These are the features used for ML")
        X = df[["VOLTS ", "TESTING AMP", "TEMPERATURE","HUMIDITY "]]
        st.write(X)
         
    with lables:    
        st.text("These are the lables used for ML")
        y = df[["Fault "]]
        st.write(y)
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model = model.fit(x_train, y_train)
    
   
    
    future, accuracy = st.columns(2)
    
    with future:
        st.subheader("ML Result")
        a = st.number_input("Input a value of Volatge" ,min_value=200, max_value=450)
        b = st.number_input("Input a value of Testing Current ", min_value=1, max_value=12)
        c = st.number_input("Input a value of Motor Temperature",min_value=0, max_value=100)
        d = st.number_input("Input a value of Motor Humidity",min_value=0, max_value=150)
        
        predictions = model.predict([[a,b,c,d]])
        st.write("This is the prediction: ",predictions)
        
    with accuracy:
        
        st.subheader("Accuracy Score Result")
        accuracy = model.score(x_test,y_test)
        st.write('Score for Training data = ', accuracy)
        
        
        
        
    
    