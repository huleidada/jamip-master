# encoding utf-8
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from .model import  RegressionModelbuilder, ClassificationModelbuider
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

class ModelfitProcessing():
    """
    Abstract class, specifically for the processing of model fit information, to 
    help identify the most useful information for the material target properties.

    Args: 
        fixed_model: class
            Models with hyperparameters already set.
    """
    __metaclass__ = ABCMeta

    def __init__(self, fixed_model):
        self._fixed_model = fixed_model

    @property
    def fixed_model(self):
       return self._fixed_model

    @abstractmethod
    def model_fit(self):

        pass

    @abstractmethod
    def model_predict(self):

        pass  

    @abstractmethod
    def model_fit_info(self):
        
        pass

class RegressionModelfitProcessing(ModelfitProcessing):
    """
    Regression model fit information.
    
    Args: 
        fixed_model: class
            Regression models with hyperparameters already set.    
    """
    def __init__(self, fixed_model):

        super(RegressionModelfitProcessing, self).__init__(fixed_model)
    
    @property
    def __get_hyper(self):
        attr = list(self.fixed_model.__dict__.keys())
        self._get_hyper = {}
        for i in range(len(attr)):
            if attr[i].endswith('_') or attr[i].startswith('_'):
                pass
            else:
                self._get_hyper[attr[i]] = self.fixed_model.__dict__[attr[i]]
        return self._get_hyper
    
    def model_fit(self, X_train, y_train):
        """
        Regression model fits
    
        Args:
            X_train: pandas.DataFrame, list, numpy
                Training set feature data, preferably in DataFrame data format.
            y_train: pandas.DataFrame or list
                Training set target property data, preferably in DataFrame data format.
        Returns: self    
        """
        self._model_fit = self.fixed_model.fit(X_train, y_train)
        return self
    
    def model_fit_cv(self, X_train, y_train, cv = 5, scoring = 'neg_root_mean_squared_error'):
        """
        Regression model cross-validation fit

        Args:
            X_train: pandas.DataFrame, list， numpy
                Training set feature data, preferably in DataFrame data format.
            y_train: pandas.DataFrame or list
                Training set target property data, preferably in DataFrame data format.
            cv = int
                Number of cross-validations.
            scoring: str
                Model evaluation metrics, see below for details:
                https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        
        Returns: float
            Cross-validation results.
        """
        self.cv_result = cross_val_score(self.fixed_model, X_train, y_train, scoring = scoring, cv = cv).mean()
        return self.cv_result

    def get_feature_importances(self, feature_name):
        """
        Obtain feature weight ranking

        Args:
            feature_name: list
                List of feature names, order to be consistent with the order under X_train.
            
        Returns: pandas.DataFrame
             Feature ranking
        """
        self._name = type(self._fixed_model).__name__
        if hasattr(self._model_fit, 'feature_importances_'):
            self._feature_importances = self._model_fit.feature_importances_
        elif hasattr(self._model_fit, 'coef_'):
            self._feature_importances = self._model_fit.coef_
        else:
            raise ValueError('The estimator has no feature weight attribute !')

        feature_dict = dict(zip(feature_name, self._feature_importances))
        feature_rank = sorted(feature_dict.items() , key = lambda x:x[1] , reverse = True)   
        for i in range(len(feature_rank)):
            feature_rank[i] = list(feature_rank[i])
        name = [self._name, 'weight_score']
        self._feature_importances_DataFrame = pd.DataFrame(columns=name, data = feature_rank)
        return self._feature_importances_DataFrame

    
    def model_fit_info(self, feature_name):
        """
        Summary of model fit information

        Args:
            feature_name: list
                List of feature names, order to be consistent with the order under X_train.
        
        Returns: self
            Generate a file containing hyperparameters, feature weights and intercept.
        """
        self._name = type(self._fixed_model).__name__
        model_info = self.__get_hyper
        if hasattr(self._model_fit, 'coef_'):
            self.get_feature_importances(feature_name)
            self._intercept = self._model_fit.intercept_
            model_info['intercept'] = self._intercept
        elif hasattr(self._model_fit, 'effective_metric_'):
            self._effective_metric = self._model_fit.effective_metric_
            self._effective_metric_params = self._model_fit.effective_metric_params_
            model_info['effective_metric'] = self._effective_metric
            model_info['effective_metric_params'] = self._effective_metric_params
        elif hasattr(self._model_fit, 'feature_importances_'):
            self.get_feature_importances(feature_name)
            model_info = model_info

        info_set = []
        if hasattr(self._model_fit, 'coef_') or hasattr(self._model_fit, 'feature_importances_'):
            self._feature_importances_DataFrame.to_csv(self._name + '_feature_importances_' + 'rank.csv')
        info_set.append(model_info)
        pd.DataFrame(info_set).to_csv(self._name +'_info.csv') 

        return self

    @property
    def intercept(self):
        if hasattr(self._model_fit, 'coef_'):
            return self._intercept
        else:
            raise ValueError('The model does not have this attribute!')

    def model_predict(self, X_test):
        """
        Regression model prediction

        Args:
            X_test: pandas.DataFrame, list, numpy
                Testing set feature data, preferably in DataFrame data format.
        
        Returns: numpy
            Return test set prediction results
        """
        self._predict_data = self._model_fit.predict(X_test)
        return self._predict_data

class ClassificationModelfitProcessing(ModelfitProcessing):
    """
    Classification model fit information.
    
    Args: 
        fixed_model: class
            Classification models with hyperparameters already set.    
    """

    def __init__(self, fixed_model):

        super(ClassificationModelfitProcessing, self).__init__(fixed_model)

    @property
    def __get_hyper(self):
        attr = list(self.fixed_model.__dict__.keys())
        self._get_hyper = {}
        for i in range(len(attr)):
            if attr[i].endswith('_') or attr[i].startswith('_'):
                pass
            else:
                self._get_hyper[attr[i]] = self.fixed_model.__dict__[attr[i]]
        return self._get_hyper

    def model_fit(self, X_train, y_train):
        """
        Classification model fits
    
        Args:
            X_train: pandas.DataFrame, list, numpy
                Training set feature data, preferably in DataFrame data format.
            y_train: pandas.DataFrame or list
                Training set target property data, preferably in DataFrame data format.
        Returns: self    
        """
        self._model_fit = self.fixed_model.fit(X_train, y_train)
        return self

    def get_feature_importances(self, feature_name):
        """
        Obtain feature weight ranking

        Args:
            feature_name: list
                List of feature names, order to be consistent with the order under X_train.
            
        Returns: pandas.DataFrame
             Feature ranking
        """
        self._name = type(self._fixed_model).__name__
        if hasattr(self._model_fit, 'feature_importances_'):
            self._feature_importances = self._model_fit.feature_importances_
        elif hasattr(self._model_fit, 'coef_'):
            self._feature_importances = self._model_fit.coef_
        else:
            raise ValueError('The estimator has no feature weight attribute ! ')

        feature_dict = dict(zip(feature_name, self._feature_importances))
        feature_rank = sorted(feature_dict.items() , key = lambda x:x[1] , reverse = True)   
        for i in range(len(feature_rank)):
            feature_rank[i] = list(feature_rank[i])
        name = [self._name, 'weight_score']
        self._feature_importances_DataFrame = pd.DataFrame(columns=name, data = feature_rank)
        return self._feature_importances_DataFrame

    
    def model_fit_info(self, feature_name):
        """
        Summary of model fit information

        Args:
            feature_name: list
                List of feature names, order to be consistent with the order under X_train.
        
        Returns: self
            Generate a file containing hyperparameters, feature weights and intercept.
        """
        self._name = type(self._fixed_model).__name__
        model_info = self.__get_hyper
        if hasattr(self._model_fit, 'coef_'):
            self.get_feature_importances(feature_name)
            self._intercept = self._model_fit.intercept_
            model_info['intercept'] = self._intercept
        elif hasattr(self._model_fit, 'effective_metric_'):
            self._effective_metric = self._model_fit.effective_metric_
            self._effective_metric_params = self._model_fit.effective_metric_params_
            model_info['effective_metric'] = self._effective_metric
            model_info['effective_metric_params'] = self._effective_metric_params
        elif hasattr(self._model_fit, 'feature_importances_'):
            self.get_feature_importances(feature_name)
            model_info = model_info

        info_set = []
        if hasattr(self._model_fit, 'coef_') or hasattr(self._model_fit, 'feature_importances_'):
            self._feature_importances_DataFrame.to_csv(self._name + '_feature_importaces_' + 'rank.csv')
        info_set.append(model_info)
        pd.DataFrame(info_set).to_csv(self._name +'_info.csv') 

        return self
        
    @property
    def feature_importances(self):
        return self._feature_importances_DataFrame

    @property
    def intercept(self):
        if hasattr(self._model_fit, 'coef_'):
            return self._intercept
        else:
            raise ValueError('The model does not have this attribute!')

    def model_predict(self, X_test):
        """
        Classification model prediction

        Args:
            X_test: pandas.DataFrame, list, numpy
                Testing set feature data, preferably in DataFrame data format.
        
        Returns: numpy
            Return test set prediction results
        """
        self._predict_data = self._model_fit.predict(X_test)
        return self._predict_data