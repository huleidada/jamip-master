# encoding utf-8
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import copy
from abc import ABCMeta, abstractmethod
import math
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_regression, f_classif, chi2, RFE, RFECV
from itertools import combinations, permutations
import warnings

class FeaturePreprocessing(object):

    __metaclass__ = ABCMeta
    """
    The abstract class defines some basic functions for feature deflation 
    and feature construction.

    Args:
        dataset_df(pandas.DataFrame): Dataframe containing all features 
            and targets
        target(str): Name of the target property to be fitted in the dataframe
        _mappingFunctions(dict): Store the mapping function for each feature

    """
    @abstractmethod
    def __init__(self, dataset_df, target):
        
        self._df = dataset_df
        self._df_original = copy.deepcopy(dataset_df)
        if target in list(self._df.columns.values):
            self.target = target
        else:
            raise ValueError(
        "The {} is not in the dataset !".format(target)
            )
        self._mappingFunctions = None

    @property
    def df_original(self):
        return self._df_original

    @property
    def original_features(self):
        return list(self.df_original.drop(columns = [self.target]).columns.values)
    
    def remove_feature(self, feature):
        """
        Remove a particular feature from a dataframe
        
        Args:
            feature(str): a feature name in the dataframe.
        """
        if self.original_features is None:
            warnings.warn('features is None')
        elif not feature in self.original_features:
            warnings.warn('not exist in features')
        else:
            self._df.pop(feature)
        return self
    
    @property
    @abstractmethod
    def mappingFunctions(self):
        """
        Store the mapping function for each feature
        """
        return self._mappingFunctions

    @abstractmethod
    def transform_x(self):
        """
        Converting processed features to new features.
        Output only the new features in dataframe format.
        """
        pass

    @abstractmethod
    def transform_dataframe(self):
        """
        Converting processed features to new features.
        Output all features in dataframe format.
        """
        pass