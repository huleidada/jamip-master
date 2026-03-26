# encoding utf-8
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from abc import ABCMeta, abstractmethod

class ModelSet():
    __metaclass__ = ABCMeta
    """
    Abstract class, this class is mainly used for machine learning model setup.
    """
    @abstractmethod
    def model_set(self, model_name):
        """
        set model name
        """
        pass

    @abstractmethod
    def hyper_set(self, *args, **kwargs):
        """
        set model hyperparameter 
        """
        pass


class RegressionModelbuilder(ModelSet):
    """
    Regression model construction

    Args:
        model_name: str
            Four classes of regression models are provided for model building, linear model, 
            nearest neighbor model, support vector machine model and tree model, 16 models in 
            total.{'LinearRegression or lr', 'BayesianRidge or bayesian', 'Lasso or lasso',
            'Ridge or rr', 'KNeighborsRegressor or knnr', 'RadiusNeighborsRegressor or rnnr', 
            'svr or SVR', 'nusvr or NuSVR', 'LinearSVR or lsvr', 'DecisionTreeRegressor or dtr',
            'ExtraTreeRegressor or etr', 'RandomForestRegressor or rfr', 'AdaBoostRegressor or abr',
            'ExtraTreesRegressor or etsr', 'GradientBoostingRegressor or gbrt'}.

    """
    def __init__(self, model_name):
        
        self.model_name = model_name
        #self.model_prototype = model_prototype
    
    @property
    def model_set(self):

        if self.model_name == 'LinearRegression' or self.model_name == 'lr':
            from sklearn.linear_model import LinearRegression
            self._model = LinearRegression
            return self
        
        elif self.model_name == 'BayesianRidge' or self.model_name == 'bayesian':
            from sklearn.linear_model import BayesianRidge
            self._model = BayesianRidge
            return self
        
        elif self.model_name == 'Lasso' or self.model_name == 'lasso':
            from sklearn.linear_model import Lasso
            self._model = Lasso
            return self

        elif self.model_name == 'Ridge' or self.model_name == 'rr':
            from sklearn.linear_model import Ridge
            self._model = Ridge
            return self

        elif  self.model_name == 'KNeighborsRegressor' or self.model_name == 'knnr':
            from sklearn.neighbors import KNeighborsRegressor
            self._model = KNeighborsRegressor
            return self
        
        elif self.model_name == 'RadiusNeighborsRegressor' or self.model_name == 'rnnr':
            from sklearn.neighbors import RadiusNeighborsRegressor
            self._model = RadiusNeighborsRegressor
            return self

        elif self.model_name == 'svr' or self.model_name == 'SVR':
            from sklearn.svm import SVR
            self._model = SVR
            return self
        
        elif self.model_name == 'nusvr' or self.model_name == 'NuSVR':
            from sklearn.svm import NuSVR
            self._model = NuSVR
            return self
        
        elif self.model_name == 'LinearSVR' or self.model_name == 'lsvr':
            from sklearn.svm import LinearSVR
            self._model = LinearSVR
            return self

        elif self.model_name == 'DecisionTreeRegressor' or self.model_name == 'dtr':
            from sklearn.tree import DecisionTreeRegressor
            self._model = DecisionTreeRegressor
            return self

        elif self.model_name == 'ExtraTreeRegressor' or self.model_name == 'etr':
            from sklearn.tree import ExtraTreeRegressor
            self._model = ExtraTreeRegressor
            return self

        elif self.model_name == 'RandomForestRegressor' or self.model_name == 'rfr':
            from sklearn.ensemble import RandomForestRegressor
            self._model = RandomForestRegressor
            return self

        elif self.model_name == 'AdaBoostRegressor' or self.model_name == 'abr':
            from sklearn.ensemble import AdaBoostRegressor
            self._model = AdaBoostRegressor
            return self

        elif self.model_name == 'ExtraTreesRegressor' or self.model_name == 'etsr':
            from sklearn.ensemble import ExtraTreesRegressor
            self._model = ExtraTreesRegressor
            return self

        elif self.model_name == 'GradientBoostingRegressor' or self.model_name == 'gbrt':
            from sklearn.ensemble import GradientBoostingRegressor
            self._model = GradientBoostingRegressor
            return self
        
        else:
            raise ValueError('Incorrect model name input!')

    def hyper_set(self, *args, **kwargs):
        """
        Set regression model parameters.
        
        Args:
            *args, **kwargs: For parameters, refer to sklearn.
        
        Return:
            Return the model with hyperparameters.
        """
        self._model_run = self._model(*args, **kwargs)
        return self._model_run


class ClassificationModelbuider(ModelSet):
    """
    Classification model construction

    Args:
        model_name: str
            Four classes of classification models are provided for model building, linear model, nearest 
            neighbor model, support vector machine model and tree model, 12 models in total.{'LogisticRegression 
            or logis', 'KNeighborsClassifier or knnc', 'RadiusNeighborsClassifier or rnc', 'SVC or svc', 'NuSVC 
            or nusvc', 'LinearSVC or lsvc', 'DecisionTreeClassifier or dtc', 'ExtraTreeClassifier or etc', 
            'RandomForestClassifier or rfc', 'AdaBoostClassifier or abc', 'ExtraTreesClassifier or etsc', 
            'GradientBoostingClassifier or gbct'}.
    """
    def __init__(self , model_name):
        
        self.model_name = model_name

    @property    
    def model_set(self):

        if self.model_name == 'LogisticRegression' or self.model_name == 'logic':
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression
            return self
        
        elif self.model_name == 'KNeighborsClassifier' or self.model_name == 'knc':
            from sklearn.neighbors import KNeighborsClassifier
            self._model = KNeighborsClassifier
            return self
        
        elif self.model_name == 'RadiusNeighborsClassifier' or self.model_name == 'rnc':
            from sklearn.neighbors import RadiusNeighborsClassifier
            self._model = RadiusNeighborsClassifier
            return self
        
        elif self.model_name == 'SVC' or self.model_name == 'svc':
            from sklearn.svm import SVC
            self._model = SVC
            return self
        elif self.model_name == 'NuSVC' or self.model_name == 'nusvc':
            from sklearn.svm import NuSVC
            self._model = NuSVC
            return self
        
        elif self.model_name == 'LinearSVC' or self.model_name == 'lsvc':
            from sklearn.svm import LinearSVC
            self._model = LinearSVC
            return self
        
        elif self.model_name == 'DecisionTreeClassifier' or self.model_name == 'dtc':
            from sklearn.tree import DecisionTreeClassifier
            self._model = DecisionTreeClassifier
            return self
        
        elif self.model_name == 'ExtraTreeClassifier' or self.model_name == 'etc':
            from sklearn.tree import ExtraTreeClassifier
            self._model = ExtraTreeClassifier
            return self
        
        elif self.model_name == 'RandomForestClassifier' or self.model_name == 'rfc':
            from sklearn.ensemble import RandomForestClassifier
            self._model = RandomForestClassifier
            return self
        
        elif self.model_name == 'AdaBoostClassifier' or self.model_name == 'abc':
            from sklearn.ensemble import AdaBoostClassifier
            self._model = AdaBoostClassifier
            return self
        
        elif self.model_name == 'ExtraTreesClassifier' or self.model_name == 'etsc':
            from sklearn.ensemble import ExtraTreesClassifier
            self._model = ExtraTreesClassifier
            return self
        

        elif self.model_name == 'GradientBoostingClassifier' or self.model_name == 'gbct':
            from sklearn.ensemble import GradientBoostingClassifier
            self._model = GradientBoostingClassifier
            return self
        
        else:
            raise ValueError('Incorrect model name input!')
    
    
    def hyper_set(self, *args, **kwargs):
        """
        Set classification model parameters.
        
        Args:
            *args, **kwargs: For parameters, refer to sklearn.
        
        Return:
            Return the model with hyperparameters.
        """

        self._model_run = self._model(*args, **kwargs)

        return self._model_run