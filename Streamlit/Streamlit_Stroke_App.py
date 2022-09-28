

# Core Pkgs
import streamlit as st 
import streamlit.components.v1 as stc
import requests 
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from predictor import predict  



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



    



        
@st.cache(allow_output_mutation=True, max_entries = 1 ) 
def get_data():
    return []

with st.form("my_form"):
   st.write("Inside the form")

   
   gender = st.selectbox('Gender',('Male', 'Female'))
   
   age = st.number_input("Age")
   
   avg_glucose_level= st.number_input("Glucose Level")
   
   hypertension = st.radio("Hypertension",('Yes', 'No',))
   if hypertension == 'Yes': 
            hypertension = 1 
   else: 
            hypertension = 0


   heart_disease = st.radio("Heart Disease",('Yes', 'No',))
   if heart_disease == 'Yes': 
            heart_disease = 1 
   else: 
            heart_disease = 0
   
   smoking_status = st.selectbox('Smoking Status',('formerly smoked', 'smokes', 'never smoked','Unknown'))

   ever_married = st.radio("Married?",('Yes', 'No',))

   work_type = st.selectbox(
                'Work Type',
                ('Private', 'Self-employed', 'Govt_job', "children"))
   
   Residence_type = st.selectbox('Residence',('Urban', 'Rural'))
    
   bmi = st.number_input("bmi Level")

   stroke = 0


# conditionals





    

    

   # Every form must have a submit button.
   
   if st.form_submit_button("Submit"):
        # Datos_Usuario = [gender, ever_married, work_type, Residence_type, smoking_status, hypertension, heart_disease, age, avg_glucose_level, bmi]
        # columns = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease", "age", "avg_glucose_level", "bmi"]
        
        
        user_data = get_data().append({"gender": gender, "ever_married": ever_married, "work_type": work_type,"Residence_type": Residence_type,"smoking_status":smoking_status,"hypertension":hypertension,"heart_disease":heart_disease,"age":age, "avg_glucose_level":avg_glucose_level, "bmi":bmi,  "stroke": stroke  })
        
        df = st.write(pd.DataFrame(get_data()))

        # def Carga_Transformer():
        #     loaded_transformer = pickle.load(open('transformer_entrenado.pkl', 'rb'))
        #     print("Cargado transformer")
        #     return loaded_transformer
        

        # def Carga_Modelo():
        #     loaded_model = pickle.load(open('modelo_entrenado.pkl', 'rb'))
        #     print(" Cargado Modelo !!!")
        #     return loaded_model

        # transformer = Carga_Transformer()
        # model = Carga_Modelo()
        # df = transformer.transform(df)
        # print(df)
        # predict= model.predict(df)
        # st.write(predict)

# @st.cache(allow_output_mutation=True)
# def get_data():
#     return []

# user_id = st.text_input("User ID")
# foo = st.slider("foo", 0, 100)
# bar = st.slider("bar", 0, 100)

# if st.button("Add row"):
#     get_data().append({"UserID": user_id, "foo": foo, "bar": bar})

# st.write(pd.DataFrame(get_data()))
