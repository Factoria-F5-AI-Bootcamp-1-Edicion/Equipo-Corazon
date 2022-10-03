from enum import auto
import streamlit as st
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from streamlit_option_menu import option_menu
import pickle
#from streamlit_option_menu import option_menu
import time
import requests


import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner


import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")
  
#Lottie Animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_hello = "https://assets7.lottiefiles.com/packages/lf20_3vbOcw.json"
lottie_url_download = "https://assets10.lottiefiles.com/packages/lf20_q56zavhf.json"
lottie_url_transition1= "https://assets10.lottiefiles.com/temp/lf20_tXDjQg.json"
lottie_url_home = "https://assets10.lottiefiles.com/packages/lf20_xgdvjjxc.json"
lottie_url_predict = "https://assets8.lottiefiles.com/temp/lf20_jbSzVz.json"
lottie_hello = load_lottieurl(lottie_url_hello)
lottie_home = load_lottieurl(lottie_url_home)
lottie_download = load_lottieurl(lottie_url_download)
lottie_transition1 = load_lottieurl(lottie_url_transition1)
lottie_predict = load_lottieurl(lottie_url_predict)

#MENU 
# Funcion para reducir el margen top
def margin(): 
    st.markdown("""
            <style>
                .css-18e3th9 {
                        padding-top: 1rem;
                        padding-bottom: 10rem;
                        padding-left: 5rem;
                        padding-right: 5rem;
                    }
                .css-1d391kg {
                        padding-top: 3.5rem;
                        padding-right: 1rem;
                        padding-bottom: 3.5rem;
                        padding-left: 1rem;
                    }
            </style>
            """, unsafe_allow_html=True)



        
#MENU 
EXAMPLE_NO = 1


def streamlit_menu(example=1):
    
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            
            selected = option_menu(
                menu_title="Menu",  # required
                options=["Home", "Info", "Prediction","About us"],  # required
                icons=["house", "heart", "clipboard-plus","chat-text"],  # optional
                #menu_icon= "cast",  # optional
                default_index=0,  # optional
                styles={
                    "menu-icon":"Data",
                    
                    "menu_title":{"font-family": "Tahoma"},
                    "nav-link": {"font-family": "Tahoma", "text-align": "left", "margin":"0px",},
                    #"nav-link-selected": {""}, 
                    })
        return selected



selected = streamlit_menu(example=EXAMPLE_NO)


if selected == "Home":
    
    st.title("Welcome to EDNA")
    st_lottie(lottie_hello, key="hello",
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
        
        height=None,
        width=None,
        ) 
    st.markdown("<h3 style='text-align: justify; color: black;'>EDNA is an artificial intelligence that can predict if a person is susceptible to suffer an stroke. You can make EDNA give a personalize prediction by filling the form a pressing a button.             Is that Simple!</h3>", unsafe_allow_html=True)
    
    

if selected == "Info":

    st.title("What is a Stroke?")

  
    

   
    st_lottie (lottie_home, key="home",
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high
    
    height=None,
    width=None,)

    st.markdown("<h3 style='text-align: justify; color: black;'>A stroke, sometimes called a brain attack, occurs when something blocks blood supply to part of the brain or when a blood vessel in the brain bursts. In either case, parts of the brain become damaged or die. A stroke can cause lasting brain damage, long-term disability, or even death.</h3>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.video("https://www.youtube.com/watch?v=AM-r6AcPsaw")



if selected== "Prediction":
    margin()

    st.title("Hello!, please fill the form to make a prediction")
    st_lottie(lottie_predict, key="predict",
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
        
        height=None,
        width=None,
        ) 



    work_type = st.selectbox(
                    'Work Type',
                    ('Private', 'Self-employed', 'Govt_job', "children"))

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


    Residence_type = st.selectbox('Residence',('Urban', 'Rural'))

    bmi = st.number_input("bmi Level")

    stroke = 0


    #def teclado():
        # Mensaje de bienvenida
        
        # Metemos datos
        
        
        
    age = int(age)
    bmi = float(bmi)
    avg_glucose_level = float(avg_glucose_level)
    heart_disease = int(heart_disease)
    hypertension = int(hypertension)
    stroke = int(stroke)
    ever_married = 'Yes'
    Residence_type = 'Urban'
        
    list_variables_predictoras = [[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke]]


    # se elimina stroke ya que es la variable objetivo
    columns = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']

    # se crea dataframe del usuario
    df_usuario_test = []
    df_usuario_test = pd.DataFrame(list_variables_predictoras, columns=columns)

    # df en crudo
    #print(df_usuario_test.head())
    #print(f"columnas", df_usuario_test.columns )
    #print (df_usuario_test)

    st.write(df_usuario_test)

    df = df_usuario_test

    df["hypertension"] = df["hypertension"].astype(bool)
    df["heart_disease"] = df["heart_disease"].astype(bool)
    df["stroke"] = df["stroke"].astype(bool)
    df["stroke"].value_counts()


    df.isnull().sum(axis = 0)


    categoricas = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease", "stroke"]
    numericas = ["age", "avg_glucose_level", "bmi"]


    ## se elimina variable objetivo, por que es la que queremos predecir
    X = df.drop("stroke", axis=1)
    y = df["stroke"]



    X.head()
    y.head()


    categoricas = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]

        
    carga_transformer = pickle.load(open('transformer_entrenado_Felix.pkl', 'rb'))    
    carga_modelo = pickle.load(open('modelo_entrenado_Felix.pkl', 'rb'))

    transformer = carga_transformer
    model = carga_modelo
    df = transformer.transform(df)

    print(f"Matriz: \n {df}")

    predict=model.predict_proba(df)
 
    
    
    ##use predict_proba instead of predict to show % in the result
   


    if st.button("Predict"):
        
            
        with st_lottie_spinner(lottie_download, key="download"):
            time.sleep(4)
            
            col1, col2, col3 = st.columns(3)
            
            col2.metric("", "%", "")
            col1.metric( value=predict[0][0],label= "Probability of suffering a stroke")
            
            # st.write("Probability of suffering a stroke", predict[0][1], "%" )
    









