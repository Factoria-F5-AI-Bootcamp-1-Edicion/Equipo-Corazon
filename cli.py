# Librerías estándar de análisis de datos
import pandas as pd
import numpy as np
import pylab 
import scipy.stats as stats

# Librerías de visualización
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")





#Mensaje de bienvenida
print("¡Hola! Introduce los datos del nuevo paciente")

#Escribimos genero
gender = input("Por favor ingrese el genero del paciente (Male/Female): ")

#Escribimos work_type
work_type = input("\nPor favor ingrese el tipo de trabajo(Private/Self-employed/Govt_job/children): \n")

##Leemos Residence_type
residence_type = input("\nPor favor ingrese el tipo de residencia(Urban/Rural): \n")

##Leemos smoking_status
smoking_status = input("\nPor favor ingrese el tipo de fumador(formerly smoked/never smoked/smokes/Unknown): \n")

##Leemos age
age = input("\nPor favor ingrese la edad del pàciente: \n")

##Leemos hypertension
hypertension = input("\nPor favor ingrese la hipertension(1 or 0): \n")

##Leemos heart_disease
heart_disease = input("\nPor favor ingrese si esta enfermo del corazón(1 or 0): \n")

##Leemos avg_glucose_level
avg_glucose_level = input("\nPor favor ingrese nivel medio de glucosa: \n")

##Leemos avg_glucose_level
bmi = input("\nPor favor ingrese el BMI (Base Muscle Index): \n")

#Age será un entero o binario (0 ó 1)
age = int(age)
#BMI, avg_glucose_level será un real, así que usamos float()
bmi = float(bmi)
avg_glucose_level = float(avg_glucose_level)
#Bool
heart_disease = int(heart_disease)
hypertension = int(hypertension)

list_variables_predictoras = [[gender, age, hypertension, heart_disease, work_type, residence_type, avg_glucose_level, bmi, smoking_status]]

list_variables_predictoras

#Llamo a mi funcion predictora
#predict(variables_predictoras)

list_variables_predictoras

columns = ['gender', 'age', 'hypertension', 'heart_disease', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

df_nuevo_test = pd.DataFrame(list_variables_predictoras, columns = columns)

print( df_nuevo_test )



# Una variable para la ruta, buenas prácticas
path_to_data = "./stroke_dataset.csv"


# Importamos el dataset
df = pd.read_csv(path_to_data)







df["hypertension"] = df["hypertension"].astype(bool)
df["heart_disease"] = df["heart_disease"].astype(bool)
df["stroke"] = df["stroke"].astype(bool)






df["stroke"].value_counts()



df.isnull().sum(axis = 0)

df_duplicadas = df[df.duplicated()]
len(df_duplicadas)


df.describe()







categoricas = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease", "stroke"]
numericas = ["age", "avg_glucose_level", "bmi"]




df.corr()








## se elimina variable objetivo, por que es la que queremos predecir
X = df.drop("stroke", axis=1)
y = df["stroke"]

print("X:\n",X)
print("y:\n",y)
#XX = df_nuevo_test
#yy = df_nuevo_test['stroke']

X.head()
y.head()

#XX.head()
#yy.head()


categoricas = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]





from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=1) # Float
# rus = RandomUnderSampler(sampling_strategy= not minority) # String
X, y = rus.fit_resample(X,y)

#XX, yy = rus.fit_resample(XX,yy)








from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer_numerico = ("transformer_numerico", MinMaxScaler(), numericas)
transformer_categorico = ("transformer_categorico", OneHotEncoder(), categoricas)

transformer = ColumnTransformer([transformer_numerico, transformer_categorico], remainder="passthrough")


transformer_numerico_ = ("transformer_numerico", MinMaxScaler(), numericas)
transformer_categorico = ("transformer_categorico", OneHotEncoder(), categoricas)

transformer = ColumnTransformer([transformer_numerico, transformer_categorico], remainder="passthrough")








X = transformer.fit_transform(X)
print(X)
#XX = transformer.fit_transform(XX)






pd.DataFrame(X, columns = transformer.get_feature_names_out())
#pd.DataFrame(XX, columns = transformer.get_feature_names_out())









from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 100)

#XX_train, XX_test, yy_train, yy_test = train_test_split(XX, yy, train_size=0.7, random_state = 100)



from sklearn.linear_model import LogisticRegression




from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

import pickle






def train_evaluate(nombre_modelo, modelo):
        
    mod = modelo()
    mod.fit(X_train, y_train)
    y_predict = mod.predict(X_test)
    

#   mod.fit(XX_train, yy_train)
#  yy_predict = mod.predict(XX_test)
    


    accuracy = accuracy_score(y_test, y_predict)
#    accuracy_y = accuracy_score(yy_test, yy_predict)
    auc = roc_auc_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    confusionmatrix = confusion_matrix(y_test, y_predict)
    
    y_pred_train = mod.predict(X_train)
#    yy_pred_train = mod.predict(XX_train)
    
    accuracy_train = accuracy_score(y_train, y_pred_train)
#    accuracy_train_y = accuracy_score(yy_train, yy_pred_train)
    auc_train = roc_auc_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train)
    confusionmatrix_train = confusion_matrix(y_train, y_pred_train)
    
    print(nombre_modelo)
    print()
    print(f"Accuracy: {accuracy}")
#    print(f"Accuracy: {accuracy_y}")
    print(f"RocAUC: {auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"ConfusionMatrix: {confusionmatrix}")

    print(nombre_modelo)
    print()
    print(f"Accuracy_train: {accuracy_train}")
#    print(f"Accuracy_train: {accuracy_train_y}")
    print(f"RocAUC_train: {auc_train}")
    print(f"Recall_train: {recall_train}")
    print(f"Precision_train: {precision_train}")
    print(f"ConfusionMatrix_train: {confusionmatrix_train}")
    print(f"\nError: ", accuracy - accuracy_train) 
    #guardar(mod)

    

train_evaluate("LogisticRegression", LogisticRegression)



def guardar(datos):
    print('Guardado !!!')
        
    file = open('modelo_entrenado.pkl', 'wb')

    pickle.dump(datos, file)
    
    file.close()
    print('\n')

def carga(datos):
    file = open('modelo_entrenado.pkl', 'rb')

    data = pickle.load(file)

    print(" Cargado !!!")
    #mod.fit(X_train, y_train)
    #y_predict = mod.predict(X_test_nuevo)
    print(data)