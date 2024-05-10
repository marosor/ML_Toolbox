import pandas as pd
import unittest

Testeo

class TestFunciones(unittest.TestCase):
    def test_describe_df (self):
        data = pd.DataFrame({
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
        resultado_esperado = pd.DataFrame({
            'dtype': ['object', 'float64', 'int64', 'int64', 'float64', 'object', 'object', 'bool', 'object', 'object', 'bool'],
            '% Missing': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'Unique': [2, 3, 2, 1, 3, 2, 2, 2, 2, 2, 2],
            '% Cardinality': [66.67, 100.0, 66.67, 33.33, 100.0, 66.67, 66.67, 66.67, 66.67, 66.67, 66.67]
        }, index=['sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'embark_town', 'alive', 'alone']).T
        resultado_obtenido = describe_df(data)
        pd.testing.assert_frame_equal(resultado_obtenido, resultado_esperado)
    
    
    
    def test_tipifica_variables (self):
        data = pd.DataFrame({
            'sex': ['male', 'female', 'female', 'male'],
            'age': [22, 38, 26, 35],
            'sibsp': [1, 1, 0, 1],
            'parch': [0, 0, 0, 0],
            'fare': [7.25, 71.2833, 7.925, 53.1],
            'class': ['Third', 'First', 'Third', 'First'],
            'who': ['man', 'woman', 'woman', 'man'],
            'adult_male': [True, False, False, True],
            'embark_town': ['Southampton', 'Cherbourg', 'Southampton', 'Southampton'],
            'alive': ['no', 'yes', 'yes', 'yes'],
            'alone': [False, False, True, False]
        })
        resultado_esperado = pd.DataFrame({
            'nombre_variable': ['sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'embark_town', 'alive', 'alone'],
            'tipo_sugerido': ['Binaria', 'Numérica Continua', 'Numérica Discreta', 'Numérica Discreta', 'Numérica Continua', 'Categórica', 'Categórica', 'Binaria', 'Categórica', 'Binaria', 'Binaria']
        })
        resultado = tipifica_variables(data)
        pd.testing.assert_frame_equal(resultado, resultado_esperado)

    
    
    def test_get_features_num_regression (self):
        data = pd.DataFrame({
        'age': [22, 38, 26, 29, 35],  
        'fare': [7.25, 71.2833, 7.925, 13.000, 35.500],  
        'sibsp': [1, 1, 0, 0, 2],  
        'parch': [0, 0, 0, 0, 2],  
        'class': [3, 1, 3, 2, 1],  
        'sex': ['male', 'female', 'female', 'male', 'female'],  
        'embark_town': ['Southampton', 'Cherbourg', 'Southampton', 'Cherbourg', 'Southampton']  
        })
        target_col = 'fare'
        umbral_corr = 0.4
        pvalue = 0.05
        resultado_esperado = ['age']
        resultado = get_features_num_regression(data, target_col, umbral_corr, pvalue)
        self.assertListEqual(resultado, resultado_esperado)

    
    
    def test_plot_features_num_regression (self):
        data = pd.DataFrame({
            'age': [22, 38, 26, 29, 35],
            'fare': [7.25, 71.2833, 7.925, 13.000, 35.500],
            'sibsp': [1, 1, 0, 0, 2],
            'parch': [0, 0, 0, 0, 2],
            'class': [3, 1, 3, 2, 1],
            'sex': ['male', 'female', 'female', 'male', 'female'],
            'embark_town': ['Southampton', 'Cherbourg', 'Southampton', 'Cherbourg', 'Southampton']
        })
        param1_target_col = 'fare'
        param2_columns = ['age', 'class', 'sibsp']
        param3_umbral_corr = 0.4
        param4_pvalue = 0.05
        resultado_esperado = ['age']
        resultado = plot_features_num_regression(data)
        pd.testing.assert_frame_equal(resultado, resultado_esperado) # compara

    
    
    def test_get_features_cat_regression (self):
        data = pd.DataFrame({
            'target': [100, 150, 200, 250, 300],  
            'gender': ['male', 'female', 'female', 'male', 'male'],  
            'class': [1, 2, 1, 3, 2],  
            'embarked': ['S', 'C', 'Q', 'S', 'C']  
        })
        param1_target_col = 'target'
        param2_pvalue = 0.05
        resultado_esperado = ['gender', 'class']
        resultado = get_features_cat_regression(data)
        pd.testing.assert_frame_equal(resultado, resultado_esperado) # compara

    
    
    def test_plot_features_cat_regression (self):
        data = pd.DataFrame({
        'fare': [72, 48, 13, 30, 60, 80, 7, 23, 52, 18],
        'gender': ['male', 'female', 'female', 'male', 'male', 'female', 'female', 'male', 'male', 'female'],
        'class': ['1st', '2nd', '1st', '3rd', '2nd', '3rd', '1st', '3rd', '2nd', '1st']
    })
        param1_target_col= "fare"
        param2_columns = ["gender", "class"]
        param3_pvalue = 0.05
        param4_with_individual_plot = False
        resultado_esperado = ['gender', 'class']
        resultado = plot_features_cat_regression(data, param1_target_col, param2_columns, param3_pvalue, param4_with_individual_plot)
        self.assertEqual(set(resultado), set(resultado_esperado)) # Comparar contenido de las listas como sets

if __name__ == '__main__':
unittest.main()

python -m unittest toolbox_ML.py