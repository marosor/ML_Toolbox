# Imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

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
    if not 0 <= umbral_corr <= 1:
        print("Error: El umbral_corr debe estar entre 0 y 1.")
        return None

    # Comprobar si pvalue, si está definido, es un número entre 0 y 1
    if pvalue is not None:
        if not 0 <= pvalue <= 1:
            print("Error: El pvalue debe estar entre 0 y 1.")
            return None
        
    # Código
    var_tip = tipifica_variables(df, 10, 20)
    col_num = var_tip[(var_tip["tipo_sugerido"] == "Numérica Continua") | (var_tip["tipo_sugerido"] == "Numérica Discreta")]["nombre_variable"].tolist()

    # Se realizan las correlaciones y se eligen las que superen el umbral
    correlaciones = df[col_num].corr()[target_col]
    columnas_filtradas = correlaciones[abs(correlaciones) > umbral_corr].index.tolist()
    if target_col in columnas_filtradas:
        columnas_filtradas.remove(target_col)
    
    # Comprobación de que si se introduce un p-value pase los tests de hipótesis
    if pvalue is not None:
        columnas_finales = []
        for col in columnas_filtradas:
            p_value_especifico = pearsonr(df[col], df[target_col])[1]
            if pvalue < (1 - p_value_especifico):
                columnas_finales.append(col)
        columnas_filtradas = columnas_finales.copy()

    return columnas_filtradas

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
    
    # Divide la lista de columnas filtradas en grupos de máximo cinco columnas
    columnas_agrupadas = [columnas_filtradas[i:i+4] for i in range(0, len(columnas_filtradas), 4)]
    
    # Generar pairplots para cada grupo de columnas
    for group in columnas_agrupadas:
        sns.pairplot(df[[target_col] + group])
        plt.show()
    
    # Devolver la lista de columnas filtradas
    return columnas_filtradas

# Función | get_features_cat_regression


# Función | plot_features_cat_regression

