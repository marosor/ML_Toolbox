# Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
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

    # Si no hay columnas válidas, retornar una lista vacía
    if not valid_columns:
        return []

    # Crear pairplots
    max_cols_per_plot = 5  # Máximo de columnas por plot
    if target_col:
        num_classes = len(dataframe[target_col].unique())
        num_plots = max(1, (num_classes + 4) // 5)  # Número de plots según EXTRA_1

        for i in range(num_plots):
            class_subset = dataframe[target_col].unique()[i*5:(i+1)*5]
            subset_df = dataframe[dataframe[target_col].isin(class_subset)]
            
            # Dividir las columnas en grupos de max_cols_per_plot según EXTRA_2
            for j in range(0, len(valid_columns), max_cols_per_plot):
                plot_columns = valid_columns[j:j+max_cols_per_plot]
                plot_columns.append(target_col)
                sns.pairplot(subset_df[plot_columns], hue=target_col)
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

