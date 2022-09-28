# Librerías estándar de análisis de datos
import pandas as pd
import pickle
#from input import teclado

# Librerías de visualización
import warnings
warnings.filterwarnings("ignore")




# Una variable para la ruta, buenas prácticas
path_to_data = "./stroke_dataset.csv"
#path_to_data = teclado()



# df One Hot Encoding
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Importamos el dataset
df = pd.read_csv(path_to_data)




df["hypertension"] = df["hypertension"].astype(bool)
df["heart_disease"] = df["heart_disease"].astype(bool)
df["stroke"] = df["stroke"].astype(bool)
df["stroke"].value_counts()



df.isnull().sum(axis = 0)
df.describe()




categoricas = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease", "stroke"]
numericas = ["age", "avg_glucose_level", "bmi"]




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
pickle.dump(transformer, open('transformer_entrenado.pkl', 'wb'))




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







def train_evaluate(nombre_modelo, modelo):
    #LogisticRegression si algo no funciona puede ser los hiperparamentros
    mod = modelo(fit_intercept=True, penalty='l2', tol=1e-5, C=0.8, solver='lbfgs', max_iter=75,warm_start=True)

    mod.fit(X_train, y_train)
    y_predict = mod.predict(X_test)
    
    print("acuracy user", accuracy_score(y_test, y_predict) )
        # Creamos el transformore

    
    



    accuracy = accuracy_score(y_test, y_predict)
    auc = roc_auc_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    confusionmatrix = confusion_matrix(y_test, y_predict)
    
    y_pred_train = mod.predict(X_train)
    
    accuracy_train = accuracy_score(y_train, y_pred_train)

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
    print(f"\nError: ", abs(accuracy - accuracy_train) )
    print()
    
    y_pred_train = mod.predict(X_train)

    
    # Creamos el modelo
    pickle.dump(mod, open('modelo_entrenado.pkl', 'wb'))
    



train_evaluate("LogisticRegression", LogisticRegression)