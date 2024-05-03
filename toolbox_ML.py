Módulo de herramientas
Este módulo debe contener el código comentado que implemente las funciones descritas y especificadas en el apartado [fuciones]. La entrega se hará en el repositorio del grupo

import pandas as pd
import unittest

Funciones dummies:

def describe_df (df):
    dataframe que tenga una columna por cada columan del dataframe original y como filas, los tipos de las columnas, el tanto por ciento de valores nulos o missings, los valores únicos y el porcentaje de cardinalidad.
    return 0

def tipifica_variables (df, entero, float):
    devuelve un dataframe con dos columnas "nombre_variable", "tipo_sugerido" que tendrá tantas filas como columnas el dataframe. En cada fila irá el nombre de una de las columnas y una sugerencia del tipo de variable
    return 0

def get_features_num_regression (df, target_col, umbral_corr float, pvalue float = None):
    devuelve una lista con las columnas numéricas del dataframe cuya correlación con la columna designada por "target_col" sea superior en valor absoluto al valor dado por "umbral_corr". Además si la variable "pvalue" es distinta de None, sólo devolvera las columnas numéricas cuya correlación supere el valor indicado y además supere el test de hipótesis con significación mayor o igual a 1-pvalue
    return 0

def plot_features_num_regression (df, target_col= "", columns list = [], umbral_corr = 0, pvalue = 0):
    Si la lista no está vacía, la función pintará una pairplot del dataframe considerando la columna designada por "target_col" y aquellas incluidas en "column" que cumplan que su correlación con "target_col" es superior en valor absoluto a "umbral_corr", y que, en el caso de ser pvalue diferente de "None", además cumplan el test de correlación para el nivel 1-pvalue de significación estadística. La función devolverá los valores de "columns" que cumplan con las condiciones anteriores.
    return 0

def get_features_cat_regression (df,target_col, pvalue = 0.05):
    una lista con las columnas categóricas del dataframe cuyo test de relación con la columna designada por 'target_col' supere en confianza estadística el test de relación que sea necesario hacer (es decir la función debe poder escoger cuál de los dos test que hemos aprendido tiene que hacer)
    return 0

def plot_features_cat_regression (df, target_col = "", columns list = [], pvalue = 0,05, with_individual_plot = False):
    la función pintará los histogramas agrupados de la variable "target_col" para cada uno de los valores de las variables categóricas incluidas en columns que cumplan que su test de relación con "target_col" es significatio para el nivel 1-pvalue de significación estadística. La función devolverá los valores de "columns" que cumplan con las condiciones anteriores
    return 0

Testeo

class TestFunciones(unittest.TestCase):
    def test_describe_df (self):
        data = pd.DataFrame ({
            'sex': ['male', 'female', 'male'],
            'age': [22.0, 38.0, 26.0],
            'sibsp': [1, 1, 0],
            'parch': [0, 0, 0],
            'fare': [7.25, 71.2833, 7.925],
            'class': ['Third', 'First', 'Third'],
            'who': ['man', 'woman', 'man'],
            'adult_male': [True, False, True],
            'embark_town': ['Southampton', 'Cherbourg', 'Southampton'],
            'alive': ['no', 'yes', 'yes'],
            'alone': [False, False, True]
        })
        resultado_esperado =  pd.DataFrame({
            'sex': ['object', 0.0, 2, 0.22],
            'age': ['float64', 0.0, 89, 9.99],
            'sibsp': ['int64', 0.0, 7, 0.79],
            'parch': ['int64', 0.0, 7, 0.79],
            'fare': ['float64', 0.0, 248, 27.83],
            'class': ['object', 0.0, 3, 0.34],
            'who': ['object', 0.0, 3, 0.34],
            'adult_male': ['bool', 0.0, 2, 0.22],
            'embark_town': ['object', 0.0, 3, 0.34],
            'alive': ['object', 0.0, 2, 0.22],
            'alone': ['bool', 0.0, 2, 0.22]
        }, index=['Tipos', '% Faltante', 'Valores Únicos', '% Cardinalidad']).T
        resultado = describe_df(data)
        pd.testing.assert_frame_equal(resultado, resultado_esperado)
    
    def test_tipifica_variables (self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3]}) # yo le pongo la info
        resultado_esperado = # yo le digo el resultado esperado
        resultado = tipifica_variables(data)
        pd.testing.assert_frame_equal(resultado, resultado_esperado) # compara

    def test_get_features_num_regression (self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3]}) # yo le pongo la info
        resultado_esperado = # yo le digo el resultado esperado
        resultado = get_features_num_regression(data)
        pd.testing.assert_frame_equal(resultado, resultado_esperado) # compara

    def test_plot_features_num_regression (self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3]}) # yo le pongo la info
        resultado_esperado = # yo le digo el resultado esperado
        resultado = plot_features_num_regression(data)
        pd.testing.assert_frame_equal(resultado, resultado_esperado) # compara

    def test_get_features_cat_regression (self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3]}) # yo le pongo la info
        resultado_esperado = # yo le digo el resultado esperado
        resultado = get_features_cat_regression(data)
        pd.testing.assert_frame_equal(resultado, resultado_esperado) # compara

    def test_plot_features_cat_regression (self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3]}) # yo le pongo la info
        resultado_esperado = # yo le digo el resultado esperado
        resultado = plot_features_cat_regression(data)
        pd.testing.assert_frame_equal(resultado, resultado_esperado) # compara

if __name__ == '__main__':
unittest.main()

python -m unittest toolbox_ML.py