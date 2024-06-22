# Imports

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from scipy import stats
from scipy.stats import pearsonr, mannwhitneyu, shapiro, ttest_ind

from bootcampviztools import *

###############################################################################

# Función | describe_df

def describe_df(df):

    """
    Función que muestra diferentes tipos de datos de un DataFrame

    Argumentos:
    df (DataFrame): DataFrame que queremos describir

    Retorna:
    DataFrame: Devuelve un DataFrame con información sobre el tipo de datos, el porcentaje de valores nulos,
    la cantidad de valores únicos y el porcentaje de cardinalidad, de todas las variables de este
    """

    # Obtener tipos de columnas
    tipos = df.dtypes

    # Calcular porcentaje de valores nulos
    porcentaje_faltante = (df.isnull().mean() * 100).round(2)

    # Obtener valores únicos
    valores_unicos = df.nunique()

    # Obtener porcentaje de cardinalidad
    porcentaje_cardinalidad = ((valores_unicos / len(df)) * 100).round(2)

    # Crear un DataFrame con la información recopilada
    resultado_describe = pd.DataFrame({
        "Tipos" : tipos,
        "% Faltante" : porcentaje_faltante,
        "Valores Únicos" : valores_unicos,
        "% Cardinalidad" : porcentaje_cardinalidad
    })

    return resultado_describe.T

###############################################################################

# Función | tipifica_variables

def tipifica_variables(df, umbral_categoria, umbral_continua):

    """
    Función que clasifica las variables del DataFrame en tipos sugeridos

    Argumentos:
    df (DataFrame): DataFrame cuyas variables queremos clasificar
    umbral_categoria (int): Umbral para considerar una variable como categórica
    umbral_continua (float): Umbral para considerar una variable como numérica continua

    Retorna:
    DataFrame: Devuelve un DataFrame con dos columnas: "nombre_variable" y "tipo_sugerido"
    """

    # Crear una lista con la tipificación sugerida para cada variable
    lista_tipifica = []

    # Iterar sobre cada columna del DataFrame
    for columna in df.columns:

        # Obtener la cardinalidad
        cardinalidad = len(df[columna].unique())

        # Determinar el tipo
        if cardinalidad == 2:
            lista_tipifica.append("Binaria")
        elif cardinalidad < umbral_categoria:
            lista_tipifica.append("Categórica")
        else:
            porcentaje_cardinalidad = (cardinalidad / len(df)) * 100
            if porcentaje_cardinalidad >= umbral_continua:
                lista_tipifica.append("Numérica Continua")
            else:
                lista_tipifica.append("Numérica Discreta")

    # Agregar el resultado a un nuevo DataFrame con los resultados
    resultado_tipifica = pd.DataFrame({"nombre_variable" : df.columns.tolist(), "tipo_sugerido" : lista_tipifica})

    return resultado_tipifica

###############################################################################

# Función | get_features_num_regression
def get_features_num_regression(df, target_col, umbral_corr, pvalue = None):

    """
    Está función debe devolver una lista con las columnas numéricas del dataframe cuya correlación con la columna designada por "target_col" 
    sea superior en valor absoluto al valor dado por "umbral_corr". Además si la variable "pvalue" es distinta de None, sólo devolverá las 
    columnas numéricas cuya correlación supere el valor indicado y además supere el test de hipótesis con significación mayor o igual a 1-pvalue.

    Argumentos:
    - df (DataFrame): un dataframe de Pandas.
    - target_col (string): el nombre de la columna del Dataframe objetivo.
    - umbral_corr (float): un valor de correlación arbitrario sobre el que se elegirá como de correlacionadas queremos que estén las columnas elegidas (por defecto 0).
    - pvalue (float): con valor "None" por defecto.

    Retorna:
    - Lista de las columnas correlacionadas que cumplen el test en caso de que se haya pasado p-value.
    """

    # Comprobaciones

    if not isinstance(df, pd.DataFrame):
        print("Error: No has introducido un DataFrame válido de pandas.")
        return None
    
    # Comprobar si target_col está en el DataFrame
    if target_col not in df.columns:
        print(f"Error: La columna {target_col} no está en el DataFrame.")
        return None
    
    # Comprobar si target_col es numérico
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna {target_col} no es numérica.")
        return None
    
    # Comprobar si umbral_corr está entre 0 y 1
    if type(umbral_corr) != float and type(umbral_corr) != int:
        print("Error: El parámetro umbral_corr", umbral_corr, " no es un número.")
    if not 0 <= umbral_corr <= 1:
        print("Error: El umbral_corr debe estar entre 0 y 1.")
        return None
    
    #Comprobar que p-value es un entero o un float, y esta en el rango [0,1]
    if pvalue is not None:
        if type(pvalue) != float and type(pvalue) != int:
            print("Error: El parámetro pvalue", pvalue, " no es un número.")
            return None
        elif  not (0 <= pvalue <= 1):
            print("Error: El parametro pvalue", pvalue, " está fuera del rango [0,1].")
            return None
        
    # Se usa la función tipifica_variables para identificar las variables numéricas
    var_tip = tipifica_variables(df, 5, 9)
    col_num = var_tip[(var_tip["tipo_sugerido"] == "Numérica Continua") | (var_tip["tipo_sugerido"] == "Numérica Discreta")]["nombre_variable"].tolist()

    # Comprobación de que hay alguna columna numérica para relacionar
    if len(col_num) == 0:
        print("Error: No hay ninguna columna númerica o discreta a analizar que cumpla con los requisitos establecidos en los umbrales.")
    else:

    # Se realizan las correlaciones y se eligen las que superen el umbral
        correlaciones = df[col_num].corr()[target_col]
        columnas_filtradas = correlaciones[abs(correlaciones) > umbral_corr].index.tolist()
        if target_col in columnas_filtradas:
            columnas_filtradas.remove(target_col)
    
        # Comprobación de que si se introduce un p-value pase los tests de hipótesis (Pearson)
        if pvalue is not None:
            columnas_finales = []
            for col in columnas_filtradas:
                p_value_especifico = pearsonr(df[col], df[target_col])[1]
                if pvalue < (1 - p_value_especifico):
                    columnas_finales.append(col)
            columnas_filtradas = columnas_finales.copy()

    if len(columnas_filtradas) == 0:
        print("No hay columna numérica que cumpla con las especificaciones de umbral de correlación y/o p-value.")
        return None

    return columnas_filtradas

###############################################################################

# Función | plot_features_num_regression
def plot_features_num_regression(df, target_col = "", columns = [], umbral_corr = 0, pvalue = None):
    
    """
    Está función pintará una pairplot del dataframe considerando la columna designada por "target_col" y aquellas 
    incluidas en "column" que cumplan que su correlación con "target_col" es superior en valor absoluto a "umbral_corr", 
    y que, en el caso de ser pvalue diferente de "None", además cumplan el test de correlación para el nivel 1-pvalue de significación estadística. 
    La función devolverá los valores de "columns" que cumplan con las condiciones anteriores.

    Argumentos:
    - df (DataFrame): un dataframe de Pandas.
    - target_col (string): el nombre de la columna del Dataframe objetivo.
    - columns (list): una lista de strings cuyo valor por defecto es la lista vacía.
    - umbral_corr (float): un valor de correlación arbitrario sobre el que se elegirá como de correlacionadas queremos que estén las columnas elegidas (por defecto 0).
    - pvalue (float): con valor "None" por defecto.

    Retorna:
    - Pairplots: columnas correlacionadas y la columna objetivo bajo nuestro criterio.
    - Lista de las columnas correlacionadas.
    """

    # Comprobaciones
    
    # Si la lista de columnas está vacía, asignar todas las variables numéricas del dataframe
    if not columns:
        columns = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
    
    # Si se proporciona una lista de columnas, comprobar si están en el DataFrame
    else:
        for col in columns:
            if col not in df.columns:
                print(f"Error: La columna {col} no está en el DataFrame.")
                return None
    
    columnas_filtradas = get_features_num_regression(df, target_col, umbral_corr, pvalue)
    
    columnas_refiltradas = []
    for col in columnas_filtradas:
        for col2 in columns:
            if col == col2:
                columnas_refiltradas.append(col)

    # Divide la lista de columnas filtradas en grupos de máximo cinco columnas
    columnas_agrupadas = [columnas_refiltradas[i:i+4] for i in range(0, len(columnas_refiltradas), 4)]
    
    # Generar pairplots para cada grupo de columnas
    for group in columnas_agrupadas:
        sns.pairplot(df[[target_col] + group])
        plt.show()
    
    # Devolver la lista de columnas filtradas
    return columnas_refiltradas

###############################################################################

# Función | get_features_cat_regression (Versión 1 - Enunciado)
def get_features_cat_regression(df, target_col, pvalue = 0.05):
    
    #Comprobar que df es un dataframe
    if not (isinstance(df, pd.DataFrame)):
        print("Error: El parámetro df", df, " no es un DataFrame.")
        return None
    
    #Comprobar que p-value es un entero o un float, y esta en el rango [0,1]
    if type(pvalue) != float and type(pvalue) != int:
        print("Error: El parámetro pvalue", pvalue, " no es un número.")
        return None
    elif  not (0 <= pvalue <= 1):
        print("Error: El parametro pvalue", pvalue, " está fuera del rango [0,1].")
        return None
        
    #Comprobar que target_col es una variable del dataframe
    if  not (target_col in df.columns):
        print("Error: El parámetro target ", target_col , " no es una columna del Dataframe.")
        return None  
      
    #Comprobar que target_col es una variable numérica
    var_tip = tipifica_variables(df, 5, 9)

    if not (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Continua") or (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Discreta"):
        print("Error: El parámetro target_col ", target_col ,"no es una columna numérica del dataframe bajo los criterios de umbrales establecidos.")
        return None

    #Hacer una lista con las colmunnas categóricas o binarias
    col_cat = var_tip[(var_tip["tipo_sugerido"] == "Categórica") | (var_tip["tipo_sugerido"] == "Binaria")]["nombre_variable"].tolist()
    if col_cat == 0:
        return None
         
    #Inicializamos la lista de salida
    col_selec = []
    
    #Por cada columna categórica o binaria
    for valor in col_cat:
        grupos = df[valor].unique()  # Obtener los valores únicos de la columna categórica
        if len(grupos) == 2:
            grupo_a = df.loc[df[valor] == grupos[0]][target_col]
            grupo_b = df.loc[df[valor] == grupos[1]][target_col]
            u_stat, p_val = mannwhitneyu(grupo_a, grupo_b)  # Aplicamos el test U de Mann
        else:
            v_cat = [df[df[valor] == grupo][target_col] for grupo in grupos] # obtenemos los grupos y los incluimos en una lista
            f_val, p_val = stats.f_oneway(*v_cat) # Aplicamos el test ANOVA. El método * (igual que cuando vimos *args hace mil años)
        if p_val < pvalue:
            col_selec.append(valor) #Si supera el test correspondiente añadimos la variable a la lista de salida

    if len(col_selec) == 0:
        print("No hay columna categórica o binaria que cumpla con las especificaciones.")
        return None
       
    return col_selec

###############################################################################

# Función | get_features_cat_regression (Versión 2)
def get_features_cat_regression_v2(df, target_col, pvalue=0.05):
    
    #Comprobar que df es un dataframe
    if not (isinstance(df, pd.DataFrame)):
        print("Error: El parámetro df", df, " no es un DataFrame.")
        return None
    
    #Comprobar que p-value es un entero o un float, y esta en el rango [0,1]
    if type(pvalue) != float and type(pvalue) != int:
        print("Error: El parámetro pvalue", pvalue, " no es un número.")
        return None
    elif  not (0 <= pvalue <= 1):
        print("Error: El parametro pvalue", pvalue, " está fuera del rango [0,1].")
        return None
        
    #Comprobar que target_col es una variable del dataframe
    if not (target_col in df.columns):
        print("Error: El parámetro target ", target_col , " no es una columna del Dataframe.")
        return None  
      
    #Comprobar que target_col es una variable numérica contínua
    var_tip = tipifica_variables(df, 5, 9)

    if not (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Continua") or (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Discreta"):
        print("Error: El parámetro target_col ", target_col , " no es una columna numérica del dataframe bajo los criterios de umbrales establecidos.")
        return None

    #Hacer una lista con las colmunnas categóricas o binarias (???)
    col_cat = var_tip[(var_tip["tipo_sugerido"] == "Categórica") | (var_tip["tipo_sugerido"] == "Binaria")]["nombre_variable"].tolist()
    if col_cat == 0:
        return None
         
    #Inicializamos la lista de salida
    col_selec = []

    #Por cada columna categórica o binaria 
    for valor in col_cat:
        grupos = df[valor].unique()  # Obtener los valores únicos de la columna categórica
        if len(grupos) == 2:
            grupo_a = df.loc[df[valor] == grupos[0]][target_col]
            grupo_b = df.loc[df[valor] == grupos[1]][target_col]
            _, p = shapiro(grupo_a) #Usamos la prueba de normalidad de Shapiro-Wilk para saber si siguen una distribución normal o no
            _, p2 = shapiro(grupo_b)
            if p < 0.05 and p2 < 0.05:
                stat, p_val = ttest_ind(grupo_a, grupo_b) # Aplicamos el t-Student si siguen una distribución normal
            else:
                u_stat, p_val = mannwhitneyu(grupo_a, grupo_b)  # Aplicamos el test U de Mann si no la siguen
        else:
            v_cat = [df[df[valor] == grupo][target_col] for grupo in grupos] # obtenemos los grupos y los incluimos en una lista
            f_val, p_val = stats.f_oneway(*v_cat) # Aplicamos el test ANOVA. El método * (igual que cuando vimos *args hace mil años)
        if p_val < pvalue:
            col_selec.append(valor) #Si supera el test correspondiente añadimos la variable a la lista de salida
        
    if len(col_selec) == 0:
        print("No hay columna categórica o binaria que cumpla con las especificaciones.")
        return None
       
    return col_selec

###############################################################################

# Función | plot_features_cat_regression
def plot_features_cat_regression(df, target_col= "", columns=[], pvalue=0.05):

    # Comprobar que df es un dataframe
    if not (isinstance(df, pd.DataFrame)):
        print("Error: El parámetro df", df, "no es un DataFrame.")
        return None
    
    # Comprobar que p-value es un entero o un float, y esta en el rango [0,1]
    if type(pvalue) != float and type(pvalue) != int:
        print("Error: El parámetro pvalue", pvalue, "no es un número.")
        return None
    elif not (0 <= pvalue <= 1):
        print("Error: El parámetro pvalue", pvalue, "está fuera del rango [0,1].")
        return None
      
    # Comprobar que target_col es una variable numérica continua
    var_tip = tipifica_variables(df, 5, 9)

    # Si no hay target_col, pedir al usuario la introducción de una
    if target_col == "":
        print("Por favor, introduce una columna objetivo con la que realizar el análisis.")
        return "plot_features_cat_regression(df, target_col= ___, ...)"

    # Comprobar que target_col es una variable del dataframe
    if  not (target_col in df.columns):
        print("Error: El parámetro target ", target_col , " no es una columna del Dataframe.")
        return None  

    if not (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Continua") or (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Discreta"):
        print("Error: El parámetro target_col ", target_col , " no es una columna numérica del dataframe bajo los criterios de umbrales establecidos.")
        return None
    

    # Si la lista de columnas está vacía, asignar todas las variables CATEGORICAS del dataframe
    if not columns:
        columns = var_tip[var_tip["tipo_sugerido"] == "Categórica"]["nombre_variable"].tolist()
    
    # Si se proporciona una lista de columnas, comprobar si están en el DataFrame
    else:
        for col in columns:
            if col not in df.columns:
                print(f"Error: La columna {col} no está en el DataFrame.")
                return None    

    df_columns = df[columns]
    df_columns[target_col] = df[target_col]        
    
    columnas_filtradas = get_features_cat_regression(df_columns, target_col, pvalue)

    # Generar los histogramas agrupados para cada columna filtrada
    for col in columnas_filtradas:        
        plot_grouped_histograms(df, cat_col=col, num_col=target_col, group_size= len(df[col].unique()))
    
    # Devolver la lista de columnas filtradas
    return columnas_filtradas

###############################################################################

# Función | plot_features_cat_regression (Versión 2)
def plot_features_cat_regression_v2(df, target_col= "", columns=[], pvalue=0.05):

    # Comprobar que df es un dataframe
    if not (isinstance(df, pd.DataFrame)):
        print("Error: El parámetro df", df, " no es un DataFrame.")
        return None
    
    # Comprobar que p-value es un entero o un float, y esta en el rango [0,1]
    if type(pvalue) != float and type(pvalue) != int:
        print("Error: El parámetro pvalue", pvalue, " no es un número.")
        return None
    elif  not (0 <= pvalue <= 1):
        print("Error: El parámetro pvalue", pvalue, " está fuera del rango [0,1].")
        return None
      
    # Comprobar que target_col es una variable numérica contínua

    var_tip = tipifica_variables(df, 5, 9)

    # Si no hay target_col, pedir al usuario la introducción de una
    if target_col == "":
        print("Por favor, introduce una columna objetivo con la que realizar el análisis.")
        return "plot_features_cat_regression(df, target_col= ___, ...)"

    # Comprobar que target_col es una variable del dataframe
    if  not (target_col in df.columns):
        print("Error: El parámetro target ", target_col , " no es una columna del Dataframe.")
        return None  

    if not (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Continua") or (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Discreta"):
        print("Error: El parámetro target_col ", target_col , " no es una columna numérica del dataframe bajo los criterios de umbrales establecidos.")
        return None

    # Si la lista de columnas está vacía, asignar todas las variables CATEGORICAS del dataframe
    if not columns:
        columns = var_tip[var_tip["tipo_sugerido"] == "Categórica"]["nombre_variable"].tolist()
    
    # Si se proporciona una lista de columnas, comprobar si están en el DataFrame
    else:
        for col in columns:
            if col not in df.columns:
                print(f"Error: La columna {col} no está en el DataFrame.")
                return None    

    df_columns = df[columns]
    df_columns[target_col] = df[target_col]       
    
    columnas_filtradas = get_features_cat_regression_v2(df_columns, target_col, pvalue)

    # Generar los histogramas agrupados para cada columna filtrada
    for col in columnas_filtradas:        
        plot_grouped_histograms(df, cat_col=col, num_col=target_col, group_size= len(df[col].unique()))
    
    # Devolver la lista de columnas filtradas
    return columnas_filtradas


#################################################################################################
#################################################################################################

# TOOLBOX II

#################################################################################################
#################################################################################################

# Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif, SelectFromModel, RFE, SequentialFeatureSelector
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import f_oneway

from Toolbox_ML import *

###############################################################################

# Función | eval_model

def eval_model(target, predicciones, tipo_de_problema, metricas):

    """
    Función que evalua un modelo de Machine Learning utilizando diferentes métricas para problemas de regresión o clasificación

    Argumentos:
    target (tipo array): Valores del target
    predicciones (tipo array): Valores predichos por el modelo
    tipo_de_problema (str): Puede ser de regresión o clasificación
    metricas (list): Lista de métricas a calcular:
                     Para problemas de regresión: "RMSE", "MAE", "MAPE", "GRAPH"
                     Para problemas de clasificación: "ACCURACY", "PRECISION", "RECALL", "CLASS_REPORT", "MATRIX", "MATRIX_RECALL", "MATRIX_PRED", "PRECISION_X", "RECALL_X"

    Retorna:
    tupla: Devuelve una tupla con los resultados de las métricas especificadas
    """

    results = []

    # Regresión
    if tipo_de_problema == "regresion":

        for metrica in metricas:
            
            if metrica == "RMSE":
                rmse = np.sqrt(mean_squared_error(target, predicciones))
                print(f"RMSE: {rmse}")
                results.append(rmse)
            
            elif metrica == "MAE":
                mae = mean_absolute_error(target, predicciones)
                print(f"MAE: {mae}")
                results.append(mae)

            elif metrica == "MAPE":
                try:
                    mape = np.mean(np.abs((target - predicciones) / target)) * 100
                    print(f"MAPE: {mape}")
                    results.append(mape)
                except ZeroDivisionError:
                    raise ValueError("No se puede calcular el MAPE cuando hay valores en el target iguales a cero")
           
            elif metrica == "GRAPH":
                plt.scatter(target, predicciones)
                plt.xlabel("Real")
                plt.ylabel("Predicción")
                plt.title("Gráfico de Dispersión: Valores reales VS Valores predichos")
                plt.show()

     # Clasificación         
    elif tipo_de_problema == "clasificacion":

        for metrica in metricas:
            
            if metrica == "ACCURACY":
                accuracy = accuracy_score(target, predicciones)
                print(f"Accuracy: {accuracy}")
                results.append(accuracy)

            elif metrica == "PRECISION":
                precision = precision_score(target, predicciones, average = "macro")
                print(f"Precision: {precision}")
                results.append(precision)

            elif metrica == "RECALL":
                recall = recall_score(target, predicciones, average = "macro")
                print(f"Recall: {recall}")
                results.append(recall)

            elif metrica == "CLASS_REPORT":
                print("Classification Report:")
                print(classification_report(target, predicciones))

            elif metrica == "MATRIX":
                print("Confusion Matrix (Absolute Values):")
                print(confusion_matrix(target, predicciones))

            elif metrica == "MATRIX_RECALL":
                cm_normalized_recall = confusion_matrix(target, predicciones, normalize = "true")
                disp = ConfusionMatrixDisplay(confusion_matrix = cm_normalized_recall)
                disp.plot()
                plt.title("Confusion Matrix (Normalized by Recall)")
                plt.show()

            elif metrica == "MATRIX_PRED":
                cm_normalized_pred = confusion_matrix(target, predicciones, normalize = "pred")
                disp = ConfusionMatrixDisplay(confusion_matrix = cm_normalized_pred)
                disp.plot()
                plt.title("Confusion Matrix (Normalized by Prediction)")
                plt.show()

            elif "PRECISION_" in metrica:
                class_label = metrica.split("_")[-1]
                try:
                    precision_class = precision_score(target, predicciones, labels = [class_label])
                    print(f"Precisión para la clase {class_label}: {precision_class}")
                    results.append(precision_class)
                except ValueError:
                    raise ValueError(f"La clase {class_label} no está presente en las predicciones")
                
            elif "RECALL_" in metrica:
                class_label = metrica.split("_")[-1]
                try:
                    recall_class = recall_score(target, predicciones, labels = [class_label])
                    print(f"Recall para la clase {class_label}: {recall_class}")
                    results.append(recall_class)
                except ValueError:
                    raise ValueError(f"La clase {class_label} no está presente en las predicciones")
                
    # Si no es regresión o clasificación
    else:
        raise ValueError("El tipo de problema debe ser de regresión o clasificación")

    return tuple(results)

###############################################################################

# Función | get_features_num_classification

def get_features_num_classification(dataframe, target_col="", columns=None, pvalue=0.05):

    """
    Selecciona las columnas numéricas de un dataframe que pasan una prueba de ANOVA frente a la columna objetivo,
    según un nivel de significación especificado.

    Argumentos:
    dataframe (pd.DataFrame): El dataframe que contiene los datos.
    target_col (str): Nombre de la columna objetivo para la clasificación. Valor por defecto es una cadena vacía.
    columns (list): Lista de nombres de columnas a considerar. Si no se proporciona, se consideran todas las columnas numéricas. Valor por defecto es None.
    pvalue (float): Nivel de significación para la prueba de ANOVA. Valor por defecto es 0.05.

    Retorna:
    list: Devuelve una lista de nombres de columnas que cumplen con el criterio de significación especificado.
    """
    
    # Validar entradas
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe debe ser un DataFrame de pandas")
    if not isinstance(target_col, str):
        raise ValueError("target_col debe ser un string")
    if columns is not None and not all(isinstance(col, str) for col in columns):
        raise ValueError("columns debe ser una lista de strings")
    if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1):
        raise ValueError("pvalue debe ser un número entre 0 y 1")
    
    # Si columns es None, igualar a las columnas numéricas del dataframe
    if columns is None:
        columns = dataframe.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filtrar solo las columnas numéricas que están en la lista
        columns = [col for col in columns if dataframe[col].dtype in ['float64', 'int64']]
    
    # Asegurarse de que target_col esté en el dataframe
    if target_col and target_col not in dataframe.columns:
        raise ValueError(f"{target_col} no está en el dataframe")
    
    # Filtrar columnas que cumplen el test de ANOVA
    valid_columns = []
    if target_col:
        unique_classes = dataframe[target_col].unique()
        for col in columns:
            groups = [dataframe[dataframe[target_col] == cls][col].dropna() for cls in unique_classes]
            if len(groups) > 1 and all(len(group) > 0 for group in groups):
                f_val, p_val = f_oneway(*groups)
                if p_val < pvalue:
                    valid_columns.append(col)
    else:
        valid_columns = columns

    return valid_columns

###############################################################################

# Función | plot_features_num_classification

def plot_features_num_classification(dataframe, target_col="", columns=None, pvalue=0.05):

    """
    Genera pairplots para visualizar la relación entre las columnas numéricas de un dataframe y una columna objetivo, 
    filtrando aquellas columnas que pasan una prueba de ANOVA según un nivel de significación especificado.

    Argumentos:
    dataframe (pd.DataFrame): El dataframe que contiene los datos.
    target_col (str): Nombre de la columna objetivo para la clasificación. Valor por defecto es una cadena vacía.
    columns (list): Lista de nombres de columnas a considerar. Si no se proporciona, se consideran todas las columnas numéricas. Valor por defecto es None.
    pvalue (float): Nivel de significación para la prueba de ANOVA. Valor por defecto es 0.05.

    Retorna:
    list: Devuelve una lista de nombres de columnas que cumplen con el criterio de significación especificado.
    """

    # Validar entradas
    if not isinstance(dataframe, pd.DataFrame):
        print("dataframe debe ser un DataFrame de pandas")
        return None
    if not isinstance(target_col, str):
        print("target_col debe ser un string")
        return None
    if columns is not None and not all(isinstance(col, str) for col in columns):
        print("columns debe ser una lista de strings")
        return None
    if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1):
        print("Eso no es un pvalue válido")
        return None

    # Si columns es None, igualar a las columnas numéricas del dataframe
    if columns is None:
        columns = dataframe.select_dtypes(include=['number']).columns.tolist()
    else:
        # Verificar si todas las columnas en la lista existen en el dataframe
        missing_columns = [col for col in columns if col not in dataframe.columns]
        if missing_columns:
            print("Esas columnas no existen en tu dataframe:", missing_columns)
            return None
        # Filtrar solo las columnas numéricas que están en la lista
        columns = [col for col in columns if dataframe[col].dtype in ['float64', 'int64']]
        if len(columns) == 0:
            print("Debes elegir columnas numéricas")
            return None

    # Asegurarse de que target_col esté en el dataframe
    if target_col and target_col not in dataframe.columns:
        print(f"Esa columna no existe en tu dataframe")
        return None

    # Filtrar columnas que cumplen el test de ANOVA
    valid_columns = []
    if target_col:
        unique_classes = dataframe[target_col].unique()
        for col in columns:
            groups = [dataframe[dataframe[target_col] == cls][col].dropna() for cls in unique_classes]
            if len(groups) > 1 and all(len(group) > 0 for group in groups):
                f_val, p_val = f_oneway(*groups)
                if p_val < pvalue:
                    valid_columns.append(col)
    else:
        valid_columns = columns

    # Si no hay columnas válidas, retornar un mensaje
    if not valid_columns:
        print("Ninguna columna cumple con el pvalue indicado")
        return []

    # Excluir la columna objetivo de los resultados
    if target_col in valid_columns:
        valid_columns.remove(target_col)

    # Crear pairplots
    max_cols_per_plot = 5  # Máximo de columnas por plot
    if target_col:
        num_classes = len(dataframe[target_col].unique())
        for i in range(0, len(valid_columns), max_cols_per_plot):
            plot_columns = valid_columns[i:i+max_cols_per_plot]
            plot_columns.append(target_col)
            sns.pairplot(dataframe[plot_columns], hue=target_col)
            plt.show()
    else:
        # Sin target_col, dividir en grupos de max_cols_per_plot
        for i in range(0, len(valid_columns), max_cols_per_plot):
            plot_columns = valid_columns[i:i+max_cols_per_plot]
            sns.pairplot(dataframe[plot_columns])
            plt.show()
    
    return valid_columns

###############################################################################

# Función | get_features_cat_classification

def get_features_cat_classification(df, target_col, normalize=False, mi_threshold=0):

    """
    La función debe devolver una lista con las columnas categóricas del dataframe cuyo valor de mutual information 
    con 'target_col' iguale o supere el valor de "mi_threshold" si el parámetro "normalize" == False. Si es True, 
    La función debe devolver una lista con las columnas categóricas del dataframe cuyo valor normalizado de mutual 
    information con 'target_col' iguale o supere el valor de "mi_threshold".

    Argumentos:
    df (pd.DataFrame): el dataframe a analizar
    target_col (columna del df): la columna del dataframe a analizar
    normalize (bool): si queremos que el valor de mutual information se normalice o no
    mi_threshold (float): el valor a superar al analizar "mutual information"

    Retorna:
    selected_columns (list): las columnas que superan el threshold impuesto
    """

    # Verificar que el dataframe es de tipo pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        print("El argumento 'df' debe ser un DataFrame.")
        return None

    # Verificar que target_col es una columna del dataframe
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no está en el DataFrame.")
        return None

    # Verificar que mi_threshold es un float
    if not isinstance(mi_threshold, float):
        print("El argumento 'mi_threshold' debe ser un valor de tipo float.")
        return None

    # Si normalize es True, verificar que mi_threshold está entre 0 y 1
    if normalize and (mi_threshold < 0 or mi_threshold > 1):
        print("El argumento 'mi_threshold' debe estar entre 0 y 1 cuando 'normalize' es True.")
        return None

    # Obtener las columnas categóricas del dataframe excluyendo la columna target
    resultado_tipifica = tipifica_variables(df, 5, 10)
    columns = resultado_tipifica[resultado_tipifica["tipo_sugerido"] == "Categórica"]["nombre_variable"].tolist()

    # Verificar que target_col es una columna categórica
    if target_col not in columns:
        print(f"La columna '{target_col}' debe ser de tipo categórico.")
        return None
    
    # Eliminar la target_col de la lista
    columns = [col for col in columns if col != target_col]

    # Calcular la información mutua
    mi_values = mutual_info_classif(df[columns], df[target_col], discrete_features=True)

    # Normalizar los valores de información mutua si normalize es True
    if normalize:
        total_mi = sum(mi_values)
        if total_mi == 0:
            print("La suma de los valores de información mutua es 0, no se puede normalizar.")
            return None
        mi_values = mi_values / total_mi

    # Filtrar las columnas que cumplen con el umbral de información mutua
    selected_columns = [col for col, mi in zip(columns, mi_values) if mi >= mi_threshold]

    return selected_columns

###############################################################################

# Función | plot_features_cat_classification

def plot_features_cat_classification(df, target_col="", columns=[], mi_threshold=0.0, normalize=False):

    """
    La función seleccionará de la lista "columns" los valores que correspondan a columnas o features categóricas del dataframe 
    cuyo valor de mutual information respecto de target_col supere el umbral puesto en "mi_threshold", con la misma consideración que 
    la función anterior, y para los valores seleccionados, pintará la distribución de etiquetas de cada valor respecto a los valores de 
    la columna "target_col". Si la lista "columns" está vacía, cogerá todas las columnas categóricas del df.

    Argumentos:
    df (pd.DataFrame): el dataframe a analizar
    target_col (str): la columna del dataframe a analizar
    columns (list): la lista de columnas a comparar con target_col
    mi_threshold (float): el valor a superar al analizar "mutual information"
    normalize (bool): si queremos que el valor de mutual information se normalice o no

    Retorna:
    selected_columns (list): las columnas que superan el threshold impuesto
    """

    # Verificar que el dataframe es de tipo pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        print("El argumento 'dataframe' debe ser un pandas DataFrame.")
        return None

    # Verificar que target_col es una columna del dataframe
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no está en el DataFrame.")
        return None

    # Verificar que mi_threshold es un float
    if not isinstance(mi_threshold, float):
        print("El argumento 'mi_threshold' debe ser un valor de tipo float.")
        return None

    # Si normalize es True, verificar que mi_threshold está entre 0 y 1
    if normalize and (mi_threshold < 0 or mi_threshold > 1):
        print("El argumento 'mi_threshold' debe estar entre 0 y 1 cuando 'normalize' es True.")
        return None

    # Si la lista está vacía, igualar columns a las variables categóricas del dataframe reusando tipifica_variables
    if not columns:
        resultado_tipifica = tipifica_variables(df, 5, 10)
        columns = resultado_tipifica[resultado_tipifica["tipo_sugerido"] == "Categórica"]["nombre_variable"].tolist()

    #Por si no hay categóricas
    if not columns:
        print("No se encontraron columnas categóricas válidas en la lista proporcionada.")
        return None
    
    # Verificar que target_col es una columna categórica
    if target_col not in columns:
        print(f"La columna '{target_col}' debe ser de tipo categórico.")
        return None
    
    #Para no analizar target_col consigo misma
    columns = [col for col in columns if col != target_col]

    # Calcular la información mutua
    mi_values = mutual_info_classif(df[columns], df[target_col], discrete_features=True)

    # Normalizar los valores de información mutua si normalize es True
    if normalize:
        total_mi = sum(mi_values)
        if total_mi == 0:
            print("La suma de los valores de información mutua es 0, no se puede normalizar.")
            return None
        mi_values = mi_values / total_mi

    # Filtrar las columnas que cumplen con el umbral de información mutua
    selected_columns = [col for col, mi in zip(columns, mi_values) if mi >= mi_threshold]

    if not selected_columns:
        print("No se encontraron columnas que superen el umbral de información mutua.")
        return None

    # Pintar la distribución de etiquetas de cada columna seleccionada respecto a target_col
    for col in selected_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col, hue=target_col)
        plt.title(f'Distribución de {col} respecto a {target_col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.legend(title=target_col)
        plt.show()

    return selected_columns

###############################################################################

# Función | super_selector

def super_selector(dataset, target_col = "", selectores = None, hard_voting = []):

    """
    Función que selecciona features de un dataframe utilizando varios métodos y realiza un hard voting entre las listas seleccionadas
    
    Argumentos:
    dataset (pd.DataFrame): DataFrame con las features y el target
    target_col (str): Columna objetivo en el dataset. Puede ser numérica o categórica
    selectores (dict): Diccionario con los métodos de selección a utilizar. Puede contener las claves "KBest", "FromModel", "RFE" y "SFS"
    hard_voting (list): Lista de features para incluir en el hard voting

    Retorna:
    dict: Diccionario con las listas de features seleccionadas por cada método y una lista final por hard voting
    """
    
    if selectores is None:
        selectores = {}
    
    features = dataset.drop(columns = [target_col]) if target_col else dataset
    target = dataset[target_col] if target_col else None
    
    result = {}

    # Caso en que selectores es vacío o None
    if target_col and target_col in dataset.columns:
        if not selectores:
            filtered_features = [col for col in features.columns if
                                 (features[col].nunique() / len(features) < 0.9999) and
                                 (features[col].nunique() > 1)]
            result["all_features"] = filtered_features

    # Aplicación de selectores si no es vacío
    if selectores:
        if "KBest" in selectores:
            k = selectores["KBest"]
            selector = SelectKBest(score_func = f_classif, k = k)
            selector.fit(features, target)
            selected_features = features.columns[selector.get_support()].tolist()
            result["KBest"] = selected_features

        if "FromModel" in selectores:
            model, threshold_or_max = selectores["FromModel"]
            if isinstance(threshold_or_max, int):
                selector = SelectFromModel(model, max_features = threshold_or_max, threshold = -np.inf)
            else:
                selector = SelectFromModel(model, threshold = threshold_or_max)
            selector.fit(features, target)
            selected_features = features.columns[selector.get_support()].tolist()
            result["FromModel"] = selected_features

        if "RFE" in selectores:
            model, n_features, step = selectores["RFE"]
            selector = RFE(model, n_features_to_select = n_features, step = step)
            selector.fit(features, target)
            selected_features = features.columns[selector.get_support()].tolist()
            result["RFE"] = selected_features

        if "SFS" in selectores:
            model, k_features = selectores["SFS"]
            sfs = SequentialFeatureSelector(model, n_features_to_select = k_features, direction = "forward")
            sfs.fit(features, target)
            selected_features = features.columns[sfs.get_support()].tolist()
            result["SFS"] = selected_features

    # Hard Voting
    if hard_voting or selectores:
        voting_features = []
        if "hard_voting" not in result:
            voting_features = hard_voting.copy()
        for key in result:
            voting_features.extend(result[key])

        feature_counts = pd.Series(voting_features).value_counts()
        hard_voting_result = feature_counts[feature_counts > 1].index.tolist()
        
        result["hard_voting"] = hard_voting_result if hard_voting_result else list(feature_counts.index)

    return result