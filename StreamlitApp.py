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
import pickle
from streamlit_option_menu import option_menu
import time
import requests


import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner


import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner


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
lottie_hello = load_lottieurl(lottie_url_hello)
lottie_download = load_lottieurl(lottie_url_download)
lottie_transition1 = load_lottieurl(lottie_url_transition1)

st.title("¡Hola! Introduce los datos del nuevo paciente")
st_lottie(lottie_hello, key="hello",
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

    
carga_transformer = pickle.load(open('transformer_entrenado.pkl', 'rb'))    
carga_modelo = pickle.load(open('modelo_entrenado.pkl', 'rb'))

transformer = carga_transformer
model = carga_modelo
df = transformer.transform(df)

print(f"Matriz: \n {df}")

predict=model.predict(df)

# if st.button("Transition"):
#     with st_lottie(lottie_transition1, key="trans1"):
#          time.sleep(4)


if st.button("Predecir"):
    
        
    with st_lottie_spinner(lottie_download, key="download"):
        time.sleep(4)
        st.metric(label="Probabilidad de Ictus", value=predict, delta="30%")
    









