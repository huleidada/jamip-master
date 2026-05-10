# encoding utf-8
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import copy
from abc import ABCMeta, abstractmethod
import math
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_regression, f_classif, chi2, RFE, RFECV
from itertools import combinations, permutations

from .base import FeaturePreprocessing

class FeatureConstruction(FeaturePreprocessing):
    """
    Feature construction is mainly about generating derived features. By derived 
    features, we mean that the original features are transformed as a function and 
    combined with features to uncover new features that can improve the predictive 
    performance of the model and are valid and reasonable.

    Args:
        dataset_df: pandas.Dataframe
            Dataframe containing all features. 
        target: str 
            Name of the target property to be fitted in the dataframe.
    """
    def __init__(self, dataset_df, target):

        super(FeatureConstruction, self).__init__(dataset_df, target)

    def add_mappingFunctions(self, feature, mappingFunction):
        if self._mappingFunctions is None:
            self._mappingFunctions = {}
        if not feature in self._mappingFunctions.keys():
            self._mappingFunctions[feature] = mappingFunction
    
    def function_transformation(self, feature, transformation_method = 'log'):
        """
        Some methods are provided in advance to perform single feature function transformations.

        Args:
            feature: str
                Selecting the features to be normalised in the dataframe.
            transformation_method: str, {'log', 'log10','exp', 'x**2', 'x**-1', 'x**0.5'}, default = 'log'
                The single-feature transformation method, which provides several common forms of 
                function transformation.
        
        Returns: self
        """
        if feature in self._df.columns.values:
            values = self._df[feature].to_numpy()

            if transformation_method == 'log':
                new_feature = 'log(' + feature + ')'
                new_values = []
                for i in range(len(values)):
                    new_values.append(getattr(math, 'log')(values[i]))
                
                self.add_mappingFunctions(feature, 'log(x)')
                self._df[new_feature] = new_values

            elif transformation_method == 'log10':
                new_feature = 'log10(' + feature + ')' 
                new_values = []
                for i in range(len(values)):
                    new_values.append(getattr(math, 'log10')(values[i]))

                self.add_mappingFunctions(feature, 'log10(x)')
                self._df[new_feature] = new_values

            elif transformation_method == 'exp':
                new_feature = 'exp(' + feature + ')'
                new_values = []
                for i in range(len(values)):
                    new_values.append(getattr(math, 'exp')(values[i]))
                
                self.add_mappingFunctions(feature, 'exp(x)')
                self._df[new_feature] = new_values

            elif transformation_method == 'x**2':
                new_feature = feature + '**2'
                values = self._df[feature].to_numpy()
                
                self.add_mappingFunctions(feature, 'x**2')
                self._df[new_feature] = eval('x**2', {'x' :values})

            elif transformation_method == 'x**-1':
                new_feature = feature + '**-1'
                values = self._df[feature].to_numpy()
                
                self.add_mappingFunctions(feature, 'x**-1')
                self._df[new_feature] = eval('x**-1', {'x':values})

            elif transformation_method == 'x**0.5':
                new_feature = feature + '**0.5'
                values = self._df[feature].to_numpy()

                self.add_mappingFunctions(feature, 'x**0.5')
                self._df[new_feature] = eval('x**0.5', {'x':values})
            else:
                raise ValueError(
            "No {} transformation method".format(transformation_method)
            )
        else:
            import warnings
            warnings.warn('This feature has existed in dataset!')
        
        return self


    def __exponential_logarithm_calculation(self, expression_of_feature):
        exp_pos, log_pos = find_exp_log_all(expression_of_feature)
        exp_formulas = []
        log_formulas = []
        for i in range(len(exp_pos)):
            if '(' not in expression_of_feature[exp_pos[i]+4:expression_of_feature.find(')',exp_pos[i])]:
                exp_formulas.append(expression_of_feature[exp_pos[i]+4:expression_of_feature.find(')',exp_pos[i])])
            elif '(' in expression_of_feature[exp_pos[i]+4:expression_of_feature.find(')',exp_pos[i])]:
                if '(' not in expression_of_feature[expression_of_feature.find(')',exp_pos[i]):
                                                    expression_of_feature.find(')', expression_of_feature.find(')',exp_pos[i]))]:
                    exp_formulas.append(expression_of_feature[exp_pos[i]+4:expression_of_feature.find(')', expression_of_feature.find(')',exp_pos[i]))])
                else:
                    raise ValueError('Please prefer to calculate the equations in brackets and replace the symbols!')

        for i in range(len(log_pos)):
            if '(' not in expression_of_feature[log_pos[i]+4:expression_of_feature.find(')',log_pos[i])]:
                log_formulas.append(expression_of_feature[log_pos[i]+4:expression_of_feature.find(')',log_pos[i])])
            elif '(' in expression_of_feature[i+4:expression_of_feature.find(')',log_pos[i])]:
                if '(' not in expression_of_feature[expression_of_feature.find(')',log_pos[i]):
                                                    expression_of_feature.find(')', expression_of_feature.find(')',log_pos[i]))]:
                    log_formulas.append(expression_of_feature[log_pos[i]+4:expression_of_feature.find(')', expression_of_feature.find(')',log_pos[i]))])
                else:
                    raise ValueError('Please prefer to calculate the equations in brackets and replace the symbols!')

        features = self.original_features

        exp_inner_values = []
        log_inner_values = []
        exp_outer_values = []
        log_outer_values = []
        for i in range(len(exp_formulas)):
            variates = {}
            for f0 in features:
                if f0 in exp_formulas[i]:
                    variates[f0] = self._df[f0]

            if variates != {}:
                exp_inner_values.append(eval(exp_formulas[i], variates))
            else:
                import warnings
                warnings.warn("{} doesn't exist in features".format(exp_formulas[i]))

            ev = []
            for j in range(len(exp_inner_values[i])):
                ev.append(getattr(math, 'exp')(exp_inner_values[i][j]))
            exp_outer_values.append(ev)

        for i in range(len(log_formulas)):
            variates = {}
            for f0 in features:
                if f0 in log_formulas[i]:
                    variates[f0] = self._df[f0]

            if variates != {}:
                log_inner_values.append(eval(log_formulas[i], variates))
            else:
                import warnings
                warnings.warn("{} doesn't exist in features".format(log_formulas[i]))
            
            lv = []
            for j in range(len(log_inner_values[i])):
                lv.append(getattr(math, 'log')(log_inner_values[i][j]))
            log_outer_values.append(lv)
        

        s_exp_formulas = []
        s_log_formulas = []
        for i in range(len(exp_formulas)):
            new_i = filter(str.isalnum, exp_formulas[i])
            s_exp_formulas.append('exp' + ''.join(list(new_i)))
        for i in range(len(log_formulas)):
            new_i = filter(str.isalnum, log_formulas[i])
            s_log_formulas.append('log' + ''.join(list(new_i)))

        for i in range(len(exp_formulas)):
            exp_formulas[i] = 'exp(' + exp_formulas[i] + ')'
        for i in range(len(log_formulas)):
            log_formulas[i] = 'log(' + log_formulas[i] + ')'

        exp_values_dict = dict(zip(s_exp_formulas, exp_outer_values))
        log_values_dict = dict(zip(s_log_formulas, log_outer_values))

        exp_values_df = pd.DataFrame(exp_values_dict)
        log_values_df = pd.DataFrame(log_values_dict)

        el_variates = {}
        for i in exp_values_df:
            el_variates[i] = exp_values_df[i]
        for i in log_values_df:
            el_variates[i] = log_values_df[i]
        
        
        for i in range(len(exp_formulas)):
            expression_of_feature = expression_of_feature.replace(exp_formulas[i], s_exp_formulas[i])
        for i in range(len(log_formulas)):
            expression_of_feature = expression_of_feature.replace(log_formulas[i], s_log_formulas[i])


        formulas = exp_formulas + log_formulas
        return el_variates, expression_of_feature, formulas  

    def feature_crosses(self, expression_of_feature):
        """
        Custom set single-feature function transformation or inter-feature arithmetic 
        operation to complete the new feature construction.

        Args：
            expression_of_feature: str
                Custom construction of new features.
                for example:
                type-1: A^2, A^3, ..., A^x, A+A^2, tanh(A), etc.
                type-2: A*B, A/B, A+B, etc

        Returns: self
        """
        if 'log' in expression_of_feature or 'exp' in expression_of_feature:
            el_variates, new_expression_of_feature, el_formulas = self.__exponential_logarithm_calculation(expression_of_feature)
           
            variates = {}
            for f0 in self.original_features:
                if f0 in new_expression_of_feature:
                    variates[f0] = self.df_original[f0]
            variates.update(el_variates)
            
            if variates != {}:
                
                self._df[expression_of_feature] = eval(new_expression_of_feature, variates)
            else:
                import warnings
                warnings.warn("{} doesn't exist in self.features".format(expression_of_feature))
        else:
            variates = {}
            for f0 in self.original_features:
                if f0 in expression_of_feature:
                    variates[f0] = self.df_original[f0]
            if variates != {}:
                self._df[expression_of_feature] = eval(expression_of_feature, variates)
            else:
                import warnings
                warnings.warn("{} doesn't exist in self.features".format(expression_of_feature))
        
        return self
    
    def transform_x(self):
        """
        Converting features that have been constructed and new features in dataframe format.

        Returns: Pandas.DataFrame
            The new feature dataframe that has been constructed.
        """
        if (list(self._df.columns.values) == list(self.df_original.columns.values)):
            raise ValueError('No feature construction!')
        
        elif (list(self._df.columns.values) != list(self.df_original.columns.values)):
            new_features = []
            for i in list(self._df.columns.values):
                if i not in list(self.df_original.columns.values):
                    new_features.append(i)
            return self._df[new_features]

    def transform_dataframe(self):
        """
        Convert features that have been constructed and output the overall dataframe.

        Returns: Pandas.DataFrame
            The overall dataframe that has been constructed.
        """
        if (list(self._df.columns.values) == list(self.df_original.columns.values)):
            raise ValueError('No feature construction!')
        
        elif (list(self._df.columns.values) != list(self.df_original.columns.values)):
            return self._df

def find_exp_log_all(str):
    exp_pos = []
    log_pos = []
    start_exp = 0
    start_log = 0
    end = len(str)
    while start_exp < end:
        i = str.find('exp', start_exp, end)
        if i == -1:
            break
        start_exp = i+1
        exp_pos.append(i)
    while start_log < end:
        j = str.find('log', start_log, end)
        if j == -1:
            break
        start_log = j+1
        log_pos.append(j)

    return exp_pos, log_pos