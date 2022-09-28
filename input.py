import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def teclado():
    # Mensaje de bienvenida
    print("¡Hola! Introduce los datos del nuevo paciente")

    # Metemos datos
    gender = input("Por favor ingrese el genero del paciente (Male/Female): ")
    work_type = input(
        "\nPor favor ingrese el tipo de trabajo(Private/Self-employed/Govt_job/children): \n")
    
    residence_type = input(
        "\nPor favor ingrese el tipo de residencia(Urban/Rural): \n")
    smoking_status = input(
        "\nPor favor ingrese el tipo de fumador(formerly smoked/never smoked/smokes/Unknown): \n")
    age = input("\nPor favor ingrese la edad del pàciente: \n")
    hypertension = input("\nPor favor ingrese la hipertension(1 or 0): \n")
    heart_disease = input(
        "\nPor favor ingrese si esta enfermo del corazón(1 or 0): \n")

    avg_glucose_level = input("\nPor favor ingrese nivel medio de glucosa: \n")
    bmi = input("\nPor favor ingrese el BMI (Base Muscle Index): \n")
    stroke = 0
    
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
    return df_usuario_test

