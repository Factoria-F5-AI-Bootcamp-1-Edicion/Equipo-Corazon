# EquipoCorazon

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

