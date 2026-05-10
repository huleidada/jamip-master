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

class FeatureScaling(FeaturePreprocessing):
    """
    Feature scaling is a method used to normalize the range of independent variables or 
    features of data. Converting an original feature into a new feature with a defined 
    range of feature. Provides four scaling methods: 'normalization'/'non_linear_normalization'/
    'standardization'/'regularization'.

    Args:
        dataset_df(pandas.DataFrame): Dataframe containing all features. 
        target(str): Name of the target property to be fitted in the dataframe.
    """
    def __init__(self, dataset_df, target):

        super(FeatureScaling,self).__init__(dataset_df, target)

    def add_mappingFunctions(self, feature, mappingFunction):
        if self._mappingFunctions is None:
            self._mappingFunctions = {}
        if not feature in self._mappingFunctions.keys():
            self._mappingFunctions[feature] = mappingFunction
        else:
            raise ValueError(
        "'{}' has been feature scaled!".format(feature)
        )

    def normalization(self, feature, normalization_method = 'min-max'):
        """
        Maps sample data into [0,1] or [-1,1] intervals so that the features belong to a uniform 
        order of magnitude. Three methods are provided to accomplish normalization:{'min-max','max-abs','mean'}

        Args:
            feature(str): Selecting the features to be normalised in the dataframe.
            normalization_method(str, default = 'min-max'): Set the normalization method, {'min-max',
                'max_abs','mean'}.'mix-max': min-max normalization, 'max-abs': maximum abs normalisation,
                'mean':mean normalization.
        
        Returns：self
        """
        if feature in self._df.columns.values:
            values = self._df[feature].to_numpy()
            min_value = np.min(values)
            max_value = np.max(values)

            if normalization_method == 'min-max':
                mapping_function = '({}-{})/({})'.format('x', min_value, max_value-min_value)
                new_feature = feature + '_normalization_min_max'
            elif normalization_method == 'max_abs':
                max_abs = np.abs(np.max(values))
                mapping_function = '({})/({})'.format('x', max_abs)
                new_feature = feature + '_normalization_max_abs'

            elif normalization_method == 'mean':
                mean_value = np.mean(values)
                mapping_function = '({}-{})/({})'.format('x', mean_value, max_value-min_value)
                new_feature = feature + '_normalization_mean'
            else:
                raise ValueError(
            "No {} normalization method".format(normalization_method)
            )
            
            self.add_mappingFunctions(feature, mapping_function)
            if normalization_method == 'max_abs':
                self._df[new_feature] = eval(mapping_function, {'x': np.abs(values)})
            else:
                self._df[new_feature] = eval(mapping_function, {'x': values})
        
        else:
            import warnings
            warnings.warn('This feature has existed in dataset!')
            
        return self
    def non_linear_normalization(self, feature, non_linear_normalization_method):
        """
        For data with a distribution that does not belong to the normal range, maps sample data 
        into [0,1] or [-1,1] intervals so that the features belong to a uniform order of magnitude.
        Three methods are provided to accomplish normalization:{'sigmoid','log10','arctan'}

        Args:
            feature(str): Selecting the features to be normalised in the dataframe.
            normalization_method(str): Set the normalization method, {'sigmoid','log10','arctan'}.
                'sigmoid': sigmoid function(1/1+e^(-x))'log10': logarithmic function(log10(x)/log10(MAX)),
                'arctan': arctan function(arctan(x)*2/pai).

        Returns: self
        """
        if feature in self._df.columns.values:
            values = self._df[feature].to_numpy()
            max_value = np.max(values)
            if non_linear_normalization_method == 'sigmoid':
                new_feature = '1/1+e^(-' + feature + ')'
                new_values = []
                for i in range(len(values)):
                    new_values.append(1/(1+math.exp(-values[i])))

                self.add_mappingFunctions(feature, '1/1+e^(-x)')
                self._df[new_feature] = new_values
            elif non_linear_normalization_method == 'log10':
                new_feature = 'log10(' + feature + ')/log10(max)'
                logmax = math.log10(max_value)
                new_values = []
                for i in range(len(values)):
                    new_values.append((math.log10(values[i]))/logmax)

                self.add_mappingFunctions(feature, 'log10(x)/log10(max)')
                self._df[new_feature] = new_values
            elif non_linear_normalization_method == 'arctan':
                new_feature = 'arctan(' + feature + ')*2/pai'
                new_values = []
                for i in range(len(values)):
                    new_values.append(math.atan(values[i])*2/np.pi)
                
                self.add_mappingFunctions(feature, 'arctan(x)*2/pai')
                self._df[new_feature] = new_values
            else:
                raise ValueError(
            "No {} non linear normalization method".format(non_linear_normalization_method)
            )
        else:
            import warnings
            warnings.warn('This feature has existed in dataset!')

        return self
    def standardization(self, feature):
        """
        Scales the data to around 0 and the distribution becomes standard normal with a mean of 0 and 
        a standard deviation of 1

        Args:
            feature(str): Selecting the features to be normalised in the dataframe.
        
        Returns: self
        """
        if feature in self._df.columns.values:
            values = self._df[feature].to_numpy()
            mean_value = np.mean(values)
            var_value = np.var(values)

            mapping_function = '({}-{})/({})'.format('x', mean_value, var_value)
            new_feature = feature + '_standardization'
        
            self.add_mappingFunctions(feature, mapping_function)
        
            self._df[new_feature] = eval(mapping_function, {'x' : values})

        else:
            import warnings
            warnings.warn('This feature has existed in dataset!')

        return self
    
    def regularization(self, feature, norm = 2):
        """
        Regularization is used to deal with collinearity, features that are highly correlated, 
        a means of removing data noise to prevent overfitting, where a penalty term is introduced 
        to constrain.

        Args:
            feature(str): Selecting the features to be normalised in the dataframe.
            norm(int, default = 2): The norm term in regularization, which is used to scale each sample 
                to the unit norm.
            
        Returns: self
        """
        if feature in self._df.columns.values:
            values = self._df[feature].to_numpy()
            norm_value = np.linalg.norm(values, ord = norm, axis = 0)

            mapping_function = '({})/({})'.format('x', norm_value)
            new_feature = feature + '_regularization'

            self.add_mappingFunctions(feature, mapping_function)

            self._df[new_feature] = eval(mapping_function, {'x':values})

        else:
            import warnings
            warnings.warn('This feature has existed in dataset!')
        
        return self

    def transform_x(self):
        """
        Converting features that have been scaled and new features in dataframe format.

        Returns: (Pandas.DataFrame)
            The new feature dataframe that has been scaled.
        """
        if self.mappingFunctions is None:
            raise ValueError('Non-existed mapping function!')
        
        else:
            new_features = []
            for i in list(self._df.columns.values):
                if i not in list(self.df_original.columns.values):
                    new_features.append(i)
            return self._df[new_features]

    def transform_dataframe(self):
        """
        Convert features that have been scaled and output the overall dataframe.

        Returns: (Pandas.DataFrame)
            The overall dataframe that has been scaled.
        """
        if self.mappingFunctions is None:
            raise ValueError('Non-existed mapping function!')
        
        elif (list(self._df.columns.values) != list(self.df_original.columns.values)):
            return self._df