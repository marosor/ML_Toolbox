# Imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from bootcampviztools import plot_categorical_numerical_relationship, plot_combined_graphs, \
    pinta_distribucion_categoricas, plot_grouped_boxplots, plot_categorical_relationship_fin, plot_grouped_histograms
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
from scipy.stats import shapiro
from scipy.stats import ttest_ind
from scipy import stats

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
      
    #Comprobar que target_col es una variable numérica contínua
    var_tip = tipifica_variables(df, 5, 9)

    if not (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Continua"):
        print("Error: El parámetro target ", target_col , " no es una columna numérica contínua del dataframe.")
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
       
    return col_selec

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
    if  not (target_col in df.columns):
        print("Error: El parámetro target ", target_col , " no es una columna del Dataframe.")
        return None  
      
    #Comprobar que target_col es una variable numérica contínua
    var_tip = tipifica_variables(df, 5, 9)

    if not (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Continua"):
        print("Error: El parametro target ", target_col , " no es una columna numérica continua del dataframe.")
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
       
    return col_selec

# Función | plot_features_cat_regression
def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):

    # Comprobar que df es un dataframe
    if not (isinstance(df, pd.DataFrame)):
        print("Error: El parámetro df", df, "no es un DataFrame.")
        return None
    
    # Comprobar que p-value es un entero o un float, y esta en el rango [0,1]
    if type(pvalue) != float and type(pvalue) != int:
        print("Error: El parámetro pvalue", pvalue, "no es un número.")
        return None
    elif  not (0 <= pvalue <= 1):
        print("Error: El parámetro pvalue", pvalue, "está fuera del rango [0,1].")
        return None
      
    # Comprobar que target_col es una variable numérica continua
    var_tip = tipifica_variables(df, 5, 9)

    # Si no hay target_col, asignar una variable numérica continua del dataframe aleatoria
    if target_col == "":
        target_cols = var_tip[var_tip["tipo_sugerido"] == "Numérica Continua"]["nombre_variable"].tolist()
        print(target_cols)
        target_col = np.random.choice(target_cols)
        print(f"La variable elegida aleatoriamente para analizar es {target_col}")
    
    # Comprobar que target_col es una variable del dataframe
    if  not (target_col in df.columns):
        print("Error: El parámetro target ", target_col , " no es una columna del Dataframe.")
        return None  

    if not (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Continua"):
        print("Error: El parametro target ", target_col , " no es una columna numérica continua del dataframe.")
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

# Función | plot_features_cat_regression_v2
def plot_features_cat_regression_v2(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):

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

    # Si no hay target_col, asignar una variable numérica continua del dataframe aleatoria
    if target_col == "":
        target_cols = var_tip[var_tip["tipo_sugerido"] == "Numérica Continua"]["nombre_variable"].tolist()
        print(target_cols)
        target_col = np.random.choice(target_cols)
        print(f"La variable elegida aleatoriamente para analizar es {target_col}")

    # Comprobar que target_col es una variable del dataframe
    if  not (target_col in df.columns):
        print("Error: El parámetro target ", target_col , " no es una columna del Dataframe.")
        return None  

    if not (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Continua"):
        print("Error: El parametro target ", target_col , " no es una columna numérica continua del dataframe.")
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