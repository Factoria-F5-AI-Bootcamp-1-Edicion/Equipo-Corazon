

# Core Pkgs
import streamlit as st 
import streamlit.components.v1 as stc
import requests 
import pandas as pd
from streamlit_option_menu import option_menu



st.title("Stroke Calculator")
EXAMPLE_NO = 1


def streamlit_menu(example=1):
    
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            
            selected = option_menu(
                menu_title="Menu",  # required
                options=["Home", "Datos", "Mapas", "Outliers", "Conclusiones"],  # required
                icons=["house", "bar-chart", "map", "exclamation-octagon","eye"],  # optional
                #menu_icon= "cast",  # optional
                default_index=0,  # optional
                styles={
                    "menu-icon":"Data",
                    
                    "menu_title":{"font-family": "Sans-serif"},
                    "nav-link": {"font-family": "Sans-serif", "text-align": "left", "margin":"0px",},
                    #"nav-link-selected": {""}, 
                    })
        return selected



selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Home":


    st.subheader("Forms Tutorial")

# Method 1: Context Manager Approach (with)
with st.container():
	firstname = st.text_input("Firstname")
	lastname = st.text_input("lastname")

    





with st.container():
    col1, col2, col3 = st.columns(3)
    
    
    
    

			





with col1:
			option = st.selectbox(
                'Gender',
                ('Male', 'Female', 'Other'))
                
                
with col2:
				dob = st.date_input("Date of Birth")

with col3:
				st.number_input("Glucose Level")
				

with st.container():
    col1, col2, col3 = st.columns(3)
    
    
    hypertension = col1.radio(
        "Hypertension",
        ('Yes', 'No',))

    with col2:
        heart_disease = col2.radio(
        "Heart Disease",
        ('Yes', 'No',))
    
    with col3:
        option = st.selectbox(
                'Smoking Status',
                ('Formerly Smoked', 'Smokes', 'Never Smoked'),key="Smoking")

st.text("")
st.text("")
st.text("")

with st.container():
    col1, col2, col3 = st.columns(3)

    married = col1.radio(
        "Married?",
        ('Yes', 'No',))
    
    with col2:
        option = st.selectbox(
                'Work Type',
                ('Private', 'Self-employed', 'Goverment', "Stay-at-home parent"))
    with col3:
        option = st.selectbox(
                'Residence',
                ('Urban', 'Rural'))


st.text("")
st.text("")
st.text("")

if st.button('Submit'):
    st.write('Why hello there')
else:
    st.write('Goodbye') 
        
    


    

        
    
