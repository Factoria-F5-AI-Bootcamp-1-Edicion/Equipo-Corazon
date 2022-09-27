# Librerías estándar de análisis de datos
import pandas as pd
import pickle
import sklearn
# Librerías de visualizaciónMale
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

# dataframe del usuario
df_usuario_test = []
df_usuario_test = pd.DataFrame(list_variables_predictoras, columns = columns)
df = df_usuario_test

print(df)

def json(df):
    # Creamos el fichero submission.csv
    df.to_json('api.json', orient='index')


def Carga_Transformer():
    loaded_transformer = pickle.load(open('transformer_entrenado.pkl', 'rb'))
    print("Cargado transformer")
    return loaded_transformer
   

def Carga_Modelo():
    loaded_model = pickle.load(open('modelo_entrenado.pkl', 'rb'))
    print(" Cargado Modelo !!!")
    return loaded_model


def Guardar_Transformer(transformer):
    print('Guardado transformer !!!')
    # Transformer 
    file = open('transformer_entrenado.pkl', 'wb')
    pickle.dump(transformer, file)
    file.close()
    print('\n')


def Guardar_Modelo(modelo_datos):
    print('Guardado modelo !!!')
    # Modelo 
    file = open('modelo_entrenado.pkl', 'wb')
    pickle.dump(modelo_datos, file)
    file.close()
    print('\n')


transformer = Carga_Transformer()
model = Carga_Modelo()
df = transformer.transform(df)
print(df)
predict=model.predict(df)
print(predict)
#Guardar_Transformer(transformer)