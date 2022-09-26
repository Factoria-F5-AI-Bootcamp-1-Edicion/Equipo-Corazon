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


X.head()
y.head()

#XX.head()
#yy.head()


categoricas = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]





from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=1) # Float
X, y = rus.fit_resample(X,y)











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

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 1)





from sklearn.linear_model import LogisticRegression




from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

import pickle


def csv():
    # Creamos el fichero submission.csv
    pres = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Transported': df_new_test['Transported']})
    pres.to_csv('submission.csv', index=False)
    pres.head()


def carga(X_test, y_test):

    loaded_model = pickle.load(open('modelo_entrenado.pkl', 'rb'))

    print(" Cargado !!!")

    result = loaded_model.score(X_test, y_test)
    print(result)



carga(X_test, y_test)


