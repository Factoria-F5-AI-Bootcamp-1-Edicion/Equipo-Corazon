# Librerías estándar de análisis de datos

import pandas as pd
import numpy as np
import scipy.stats as stats

# Librerías de visualización
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")







# Una variable para la ruta, buenas prácticas
path_to_data = "./stroke_dataset.csv"

# variables  data usuario

#Mensaje de bienvenida

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

# dataframe del usuario
df_usuario_test = []
df_usuario_test = pd.DataFrame(list_variables_predictoras, columns = columns)

# df en crudo
print( df_usuario_test.head() )
#print(f"columnas", df_usuario_test.columns )
print()
print()


# df One Hot Encoding
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_usuario_test = pd.get_dummies(df_usuario_test)
print(f"\n----------------dummies---------------\n", df_usuario_test.head() )

#df_usuario_test = pd.DataFrame(scaler.fit_transform(df_usuario_test))
#print(f"\n-----------------test-----------------\n", df_usuario_test.head() )


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

#print("X:\n",X)
#print("y:\n",y)


X.head()
y.head()

#XX.head()
#yy.head()


categoricas = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]





from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=1) # Float
X, y = rus.fit_resample(X,y)
#from imblearn import under_sampling
#balanced = under_sampling.NearMiss()
#X, y = balanced.fit_resample(X, y)


#//who to connect a sqlite3?







from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer_numerico = ("transformer_numerico", MinMaxScaler(), numericas)
transformer_categorico = ("transformer_categorico", OneHotEncoder(), categoricas)

transformer = ColumnTransformer([transformer_numerico, transformer_categorico], remainder="passthrough")










X = transformer.fit_transform(X)
print(X)










pd.DataFrame(X, columns = transformer.get_feature_names_out())
#pd.DataFrame(XX, columns = transformer.get_feature_names_out())









from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)






from sklearn.linear_model import LogisticRegression




from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

import pickle





def train_evaluate(nombre_modelo, modelo):
    #LogisticRegression si algo no funciona puede ser los hiperparamentros
    mod = modelo(fit_intercept=True, penalty='l2', tol=1e-5, C=0.8, solver='lbfgs', max_iter=75,warm_start=True)

#    csv(mod)
#    json(mod)

#    return carga(X_test, y_test)
    

    mod.fit(X_train, y_train)
    y_predict = mod.predict(X_test)
    
    mod.fit(X_train, y_train)
    user = mod.predict(df_usuario_test)
    print("acuracy user", accuracy_score(y_test, y_predict) )
    


    accuracy = accuracy_score(y_test, y_predict)
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
    print()
    y_pred_train = mod.predict(X_train)
    print(user )




train_evaluate("LogisticRegression", LogisticRegression)


########  ESTO O SE EJECUTA #############

def guardar(datos):
    print('Guardado !!!')
        
    file = open('modelo_entrenado.pkl', 'wb')

    pickle.dump(datos, file)
    
    file.close()
    print('\n')



def carga(X_test, y_test):

    loaded_model = pickle.load(open('modelo_entrenado.pkl', 'rb'))

    print(" Cargado !!!")

    result = loaded_model.score(X_test, y_test)
    print(result)


def json(mod):
    df.to_json('api.json', orient='index')


def csv(mod): 
    df_new_test = df.drop(columns =["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"],axis = 1)
    print( df_new_test.head() )

    # One Hot Encoding
    df_new_test = pd.get_dummies(df_new_test,columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"])
    print( df_new_test.head() )

    df_new_test[["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]] = pd.DataFrame(scaler.fit_transform(df_new_test[["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]]))

    print( df_new_test.head() )
    
    #predicion para Transported para el test de datos 
    # ADB es la variable del modelo Ada Boost Classifier/ rfc randon fores
    df_new_test['stroke'] = mod.predict(df_new_test)
    
    print( df_new_test.head() )
    
    # Creamos el fichero submission.csv
    pres = pd.DataFrame({'PassengerId':df['PassengerId'],'Transported': df_new_test['Transported']})
    pres.to_csv('submission.csv', index=False)
    print( pres.head() )

