# encoding utf-8
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import copy
from abc import ABCMeta, abstractmethod
from ..model import RegressionModelbuilder, ClassificationModelbuider
from ..model import RegressionModelfitProcessing, ClassificationModelfitProcessing
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_regression, f_classif, chi2, RFE, RFECV
import matplotlib.pyplot as plt

class FeatureSelection(object):
    
    __metaclass__ = ABCMeta
    """"
    Feature selection is used to eliminate irrelevant or redundant features, 
    reduce the number of invalid features by mapping or transforming features 
    from high-dimensional space to low-dimensional space, uncover a number of 
    features that are most closely related to the material target properties 
    and identify the physical mechanisms behind them, thereby reducing model 
    training time and improving model accuracy.

    Args:
        dataset_df: pandas.DataFrame
            Dataframe containing all features and target.
        target: str 
            Name of the target property to be fitted in the dataframe.
        learning_task: str, {'regression','classification'}
            Machine learning algorithms are divided into regression and classification 
            tasks according to the category of learning tasks.
    """
    @abstractmethod
    def __init__(self, dataset_df, target, learning_task):

        self._df = dataset_df
        self._df_original = copy.deepcopy(dataset_df)
        if target in list(self._df.columns.values):
            self.target = target
        else:
            raise ValueError(
        "The {} is not in the dataset !".format(target)
            )
        self.learning_task = learning_task
    
    @property
    def df_original(self):
        return self._df_original
    
    @property
    def original_features(self):
        return list(self.df_original.drop(columns = [self.target]).columns.values)

    @property
    def features_df(self):
        return self.df_original[self.original_features]

    @property
    def target_df(self):
        return self.df_original[self.target]

    @abstractmethod
    def get_selected_features(self, *args, **kwargs):

        pass

    @abstractmethod
    def fit(self, *args, **kwargs):

        pass



class TreeBasedFeatureSelection(FeatureSelection):
    """
    Enbedded method for feature selection, feature selection work based on tree models in sklearn 
    that have the .feature_importances_ attribute.
    Args:
        dataset_df: pandas.DataFrame
            Dataframe containing all features and target.
        target: str 
            Name of the target property to be fitted in the dataframe.
        learning_task: str, {'regression','classification'}
            Machine learning algorithms are divided into regression and classification 
            tasks according to the category of learning tasks.
        model: str, {'rfr','RandomForestRegressor','RF','gbrt', 'GradientBoostingRegressor','GBRT',
                    'etsr', 'ExtraTreesRegressor', 'ESRT','rfc', 'RandomForestClassifier', 
                    'gbct', 'GradientBoostingClassifier', 'GBCT', 'etsc', 'ExtraTreesClassifier', 'ETSC'}
            Select a tree model for feature selection that is consistent with the 'learning task'.
        importance_percentile_threshold: float, default = 0.95
            Select features based on a percentage threshold in descending order of feature importance
    """
    def __init__(self, dataset_df, target, learning_task, model, importance_percentile_threshold = 0.95):
        
        super(TreeBasedFeatureSelection, self).__init__(dataset_df, target, learning_task)
        if importance_percentile_threshold > 1:
            raise ValueError('Importance percentile threshold must be lower than 1!')
        else:
            self.model = model
            if importance_percentile_threshold <= 1:
                self.importance_percentile_threshold = importance_percentile_threshold   #float
            elif importance_percentile_threshold > 1:
                raise ValueError('Please enter a number less than 1!')  
    
    def __selected_features(self, feature_rank):
        feature_ranks = feature_rank.values.tolist()
        selected_features = []
        percentile = 0
        for i in range(len(feature_ranks)):
            selected_features.append(feature_ranks[i][0])
            percentile += feature_ranks[i][1]
            if percentile >= self.importance_percentile_threshold:
                break
        return selected_features

    def set_model_hyper(self, *args, **kwargs):
        """
        Set the parameters of the selected tree model.

        Returns: self
        """
        if self.learning_task == 'regression':
            if self.model in ['rfr', 'RandomForestRegressor','RF']:
                self._tree = RegressionModelbuilder('RandomForestRegressor').model_set.hyper_set(*args, **kwargs)
            elif self.model in ['gbrt', 'GradientBoostingRegressor', 'GBRT']:
                self._tree = RegressionModelbuilder('GradientBoostingRegressor').model_set.hyper_set(*args, **kwargs)
            elif self.model in ['etsr', 'ExtraTreesRegressor', 'ESRT']:
                self._tree = RegressionModelbuilder('ExtraTreesRegressor').model_set.hyper_set(*args, **kwargs)
            else:
                raise ValueError("Unsupported {} model !".format(self.model))
        elif self.learning_task == 'classification':
            if self.model in ['rfc', 'RandomForestClassifier','RF']:
                self._tree = ClassificationModelbuider('RandomForestClassifier').model_set.hyper_set(*args, **kwargs)
            elif self.model in ['gbct', 'GradientBoostingClassifier', 'GBCT']:
                self._tree = ClassificationModelbuider('GradientBoostingClassifier').model_set.hyper_set(*args, **kwargs)
            elif self.model in ['etsc', 'ExtraTreesClassifier', 'ETSC']:
                self._tree = ClassificationModelbuider('ExtraTreesClassifier').model_set.hyper_set(*args, **kwargs)
            else:
                raise ValueError("Unsupported {} model !".format(self.model))
        return self
    
    @property
    def tree(self):
        return self._tree

    def fit(self):
        """
        Fit the features to the target selection to obtain the minimum number of features required.

        Returns: self
        """
        if self.learning_task == 'regression':
            self._fea_ranks = RegressionModelfitProcessing(self.tree).model_fit(self.features_df, self.target_df).get_feature_importances(self.original_features)
        elif self.learning_task == 'classification' :
            self._fea_ranks = ClassificationModelfitProcessing(self.tree).model_fit(self.features_df,self.target_df).get_feature_importances(self.original_features)
        
        self._final_features = self.__selected_features(self._fea_ranks)
        self._selected_dataframe =  self.features_df[self._final_features]
        return self

    @property
    def get_selected_features(self):
        """
        Get the selected features in list form.

        Returns: list
            selected features
        """
        return self._final_features

    @property
    def get_selected_dataframe(self):
        """
        Get the selected features in dataframe form.

        Returns: pandas.DataFrame
            selected features
        """
        return self._selected_dataframe

    @property 
    def features_rank(self):
        """
        Get the feature ranking of the original features

        Returns: pandas.DataFrame
            Feature ranking
        """
        return self._fea_ranks

class RegularizationFeatureSelection(FeatureSelection):

    """
    Embedded method for feature selection, feature selection work based on Regularized linear model 
    in sklearn that have the .coef_ attribute for regression learning tasks.

    Args:
        dataset_df: pandas.DataFrame
            Dataframe containing all features and target.
        target: str 
            Name of the target property to be fitted in the dataframe.
        model: str, {'l1','L1','l2','L2'}
            Select a regularized linear model as the feature selection method.
        coef_selection_threshold: float
            Set coefficient threshold for feature and target properties.
    """
    def __init__(self, dataset_df, target, model, coef_selection_threshold , learning_task = 'regression'):
        
        super(RegularizationFeatureSelection, self).__init__(dataset_df, target, learning_task)
        self.model = model
        self.coef_selection_threshold = coef_selection_threshold
        if self.learning_task != 'regression':
            raise ValueError('For regression learning tasks only!')

    def __selected_features(self, feature_rank):
        feature_ranks = feature_rank.values.tolist()
        selected_features = []
        for i in range(len(feature_ranks)):
            if abs(feature_ranks[i][1]) > self.coef_selection_threshold:
                selected_features.append(feature_ranks[i][0])
        return selected_features

    def set_model_hyper(self, *args, **kwargs):
        """
        Set the parameters of the selected regularized linear model.

        Returns: self
        """
        if self.model == 'l1' or self.model == 'L1':
            self._linear_reg = RegressionModelbuilder('Lasso').model_set.hyper_set( *args, **kwargs)
        elif self.model == 'l2' or self.model == 'L2':
            self._linear_reg = RegressionModelbuilder('Ridge').model_set.hyper_set( *args, **kwargs)
        return self

    @property
    def linear_reg(self):
        return self._linear_reg

    def fit(self):
        """
        Fit the features to the target selection to obtain the minimum number of features required.

        Returns: self
        """
        self._fea_ranks = RegressionModelfitProcessing(self.linear_reg).model_fit(self.features_df, self.target_df).get_feature_importances(self.original_features)
        self._final_features = self.__selected_features(self._fea_ranks)
        self._selected_dataframe = self.features_df[self._final_features]
        return self
    
    @property
    def get_selected_features(self):
        """
        Get the selected features in list form.

        Returns: list
            selected features
        """
        return self._final_features

    @property
    def get_selected_dataframe(self):
        """
        Get the selected features in dataframe form.

        Returns: pandas.DataFrame
            selected features
        """
        return self._selected_dataframe

    @property
    def features_rank(self):
        """
        Get the feature ranking of the original features

        Returns: pandas.DataFrame
            Feature ranking
        """
        return self._fea_ranks

class FilterFeatureSelection(FeatureSelection):
    
    """
    Filter method for feature selection, based on mutual information, correlation coefficient 
    and chi-square method in sklearn to obtain the minimum number of features required.
    
    Args:
        dataset_df: pandas.DataFrame
            Dataframe containing all features and target.
        target: str 
            Name of the target property to be fitted in the dataframe.
        learning_task: str, {'regression','classification'}
            Machine learning algorithms are divided into regression and classification 
            tasks according to the category of learning tasks.
        selected_threshold: int or float
            The number of features after filtering can be output as int or float number, int 
            type must be less than the number of features, float type must be less than 1.
    """
    def __init__(self, dataset_df, target, learning_task, selected_threshold):
        
        super(FilterFeatureSelection, self).__init__(dataset_df, target, learning_task)
        #self.filter_function = filter_function
        self.selected_threshold = selected_threshold

    @property
    def selected_feature_num(self):
        if isinstance(self.selected_threshold,int):
            if self.selected_threshold < len(self.original_features): 
                self._selected_feature_num = self.selected_threshold
            else:
                raise ValueError('The selection threshold must be less than feature number !')
        elif isinstance(self.selected_threshold, float): 
            if self.selected_threshold < 1:
                self._selected_feature_num = self.selected_threshold * self.original_features
            else:
                raise ValueError('The selection threshold must be less than 1 !')
        return self._selected_feature_num

    def __selected_features(self, feature_ranks):
        
        #feature_ranks = feature_ranks.values.tolist()
        selected_features = []
        a = 0
        for i in range(len(feature_ranks)):
            a += 1
            if a <= self.selected_feature_num:
                selected_features.append(feature_ranks[i][0])
        return selected_features

    def set_filter_function(self, filter_function, *args, **kwargs):
        """
        Set the parameters of the selected filter model.

        filter_function: str, {'mutual_information','F','chi'}
            Filter method has different methods for different learning tasks, for regression 
            task, you can choose {'mutual_information','F'}, for classification task, you 
            can choose {'mutual_information','F','chi'}.
            
        Returns: self
        """
        if self.learning_task == 'regression':
            if filter_function == 'mutual_information':
                self._filter = mutual_info_regression(self.features_df, self.target_df, discrete_features= 'auto', *args, **kwargs)
            elif filter_function == 'F':
                self._filter = f_regression(self.features_df , self.target_df, *args, **kwargs)[0]
            elif filter_function not in ['mutual_information','F']:
                raise ValueError("Unsupported {} model !".format(filter_function))

        elif self.learning_task == 'classification':
            if filter_function == 'maximal_information':
                self._filter = mutual_info_classif(self.features_df, self.target_df, discrete_features= 'auto', *args, **kwargs)
            elif filter_function == 'F':
                self._filter = f_classif(self.features_df, self.target_df, *args, **kwargs)[0]
            elif filter_function == 'chi':
                self._filter = chi2(self.features_df, self.target_df)
            elif filter_function not in ['maximal_information', 'F','chi']:
                raise ValueError("Unsupported {} model !".format(filter_function))
        self.filter_function = filter_function
        return self

    @property
    def filter(self):
        return self._filter

    def fit(self):
        """
        Fit the features to obtain the minimum number of features required.

        Returns: self
        """
        self._fea_ranks = sorted(dict(zip(self.original_features, self.filter)).items(), 
                                key = lambda x:x[1], 
                                reverse = True)
        
        self._final_features = self.__selected_features(self._fea_ranks)
        self._selected_dataframe = self.features_df[self._final_features]
        return self

    @property
    def get_selected_features(self):
        """
        Get the selected features in list form.

        Returns: list
            selected features
        """
        return self._final_features

    @property
    def get_selected_dataframe(self):
        """
        Get the selected features in dataframe form.

        Returns: pandas.DataFrame
            selected features
        """
        return self._selected_dataframe

    @property
    def features_rank(self):
        """
        Get the feature ranking of the original features

        Returns: pandas.DataFrame
            Feature ranking
        """
        for i in range(len(self._fea_ranks)):
            self._fea_ranks[i] = list(self._fea_ranks[i])
        name = [self.filter_function, 'score']
        return pd.DataFrame(columns=name, data = self._fea_ranks)

class RecursiveFeatureSelection(FeatureSelection):
    """
    Wrapper method for feature selection, use a base model to iterate through the training process 
    to select the best features set.

    Args:
        model：str
            Base model name, optionally model reference model.py
        dataset_df: pandas.DataFrame
            Dataframe containing all features and target.
        target: str
            Name of the target property to be fitted in the dataframe.
        learning_task: str
            Machine learning algorithms are divided into regression and classification 
            tasks according to the category of learning tasks.
        selected_min_features: int 
            Minimum number of features selected.
        scoring: None, str, default = None
            This parameter is considered in conjunction with 'cv'. If you only want to do 
            recursive feature search without considering the whole process, scoring is 
            set to 'None'; if you want to do cross-validation, you need to set both 'cv' 
            and 'scoring', with scoring considering a criterion, eg: 'neg_root_mean_squared_error'.
        cv: False, int, default = False
            Cross-validation, if set to 'False', means no cross-validation while scoring is set to 
            'None', if set to 'int' means cross-validation.
        step: int
            Step size for recursive feature elimination.
    """
    def __init__(self, model, dataset_df, target, learning_task, selected_min_features, scoring = None, cv = False ,step = 1):
        
        super(RecursiveFeatureSelection, self).__init__(dataset_df, target, learning_task)
        
        self.model = model
        self.selected_min_features = selected_min_features
        self.scoring = scoring
        self.cv = cv 
        self.step = step

    def set_model_hyper(self, *args, **kwargs):
        if self.learning_task == 'regression':
            self._estimator = RegressionModelbuilder(self.model).model_set.hyper_set(*args, **kwargs)
        elif self.learning_task == 'classification' or 'classifier':
            self._estimator = ClassificationModelbuider(self.model).model_set.hyper_set(*args, **kwargs)
        return self

    def set_selector(self):
        """
        Choose between 'RFE' or 'RFECV' for recursive feature elimination.
        """
        if isinstance(self.cv, bool):
            if self.cv == True:
                raise TypeError('For cross-validation enter int numbers, not True!')
            elif isinstance(self.scoring, str):
                raise ValueError("RFE cannot set 'scoring' parameters !")
            self._selector = RFE(self._estimator, 
                                 n_features_to_select = self.selected_min_features,
                                 step = self.step,)

        elif isinstance(self.cv, int):
            if self.scoring is None:
                raise ValueError("Please set the 'scoring' parameter !")
            self._selector = RFECV(self._estimator,
                                   min_features_to_select = self.selected_min_features,
                                   step = self.step,
                                   cv = self.cv,
                                   scoring = self.scoring,)                           
        return self

    @property
    def selector(self):
        return self._selector
    
    def __selected_features(self, feature_ranks):
        
        selected_features = []
        for i in range(self.selected_min_features):
            selected_features.append(feature_ranks[i][0])
        return selected_features

    def fit(self):
        """
        Fit the features to obtain the minimum number of features required.

        Returns: self
        """
        self._fit = self.selector.fit(self.features_df, self.target_df)
        self._fea_ranks = sorted(dict(zip(self.original_features, self.selector.ranking_)).items(), 
                                key = lambda x:x[1], 
                                reverse = False) 
        self._final_features = self.__selected_features(self._fea_ranks)
        self._selected_dataframe = self.features_df[self._final_features]
        if isinstance(self.scoring, str):
            self._scores = self._selector.grid_scores_
        return self

    @property
    def get_selected_features(self):
        """
        Get the selected features in list form.

        Returns: list
            selected features
        """
        return self._final_features

    @property
    def get_selected_dataframe(self):
        """
        Get the selected features in dataframe form.

        Returns: pandas.DataFrame
            selected features
        """
        return self._selected_dataframe

    @property
    def features_rank(self):
        """
        Get the feature ranking of the original features

        Returns: pandas.DataFrame
            Feature ranking
        """
        for i in range(len(self._fea_ranks)):
            self._fea_ranks[i] = list(self._fea_ranks[i])
        name = [self.model, 'rank']
        return pd.DataFrame(columns=name, data = self._fea_ranks)

    @property
    def get_scores(self):
        """
        Get the score for each iteration of the recursive feature elimination process for RFECV.
        
        Returns: pandas.DataFrame
            Iterations number and score.
        """
        self._scores_rank = []
        for i in range(len(self._scores)):
            z = []
            z.append(self._scores[i])
            z.append(i+1)
            self._scores_rank.append(z)
        return pd.DataFrame(columns = ['score','Iterations number' ], data = self._scores_rank)

    def plot_rfe_cv(self, file_name, file_format = 'jpg', linewidth = 1.5, dpi = 600, grid = True, grid_linestyle = '--'):
        """ 
        Plot the entire RFECV recursive feature removal process.
        
        Args:
            file_name: str
                Image name.
            file_format: str,default = 'jpg'
                File format, {'svg','jpg','png',...}
            linewidth: float or int, default = 1.5
                Image edge width.
            dpi: int, default = 600
                Image resolution.
            grid: bool, default = True
                Whether the network is represented in the image.
            grid_linestyle: str, default = '--'
                Network form.
        
        Returns： fig
        """
        if isinstance(self.cv, bool):
            raise ValueError("RFE has no attribute 'plot_rfe_cv'")
        else:
            font = {
                  'weight' : 'semibold',
                  'style' : 'normal',
                  'size' : 10
                    }
            label = self.target
            fig, ax = plt.subplots(nrows = 1, ncols = 1)
            ax = plt.subplot(111)
            
            ax.plot(range(len(self._scores)),
                 self._scores,
                 marker = 'o', 
                 markersize = 6, 
                 mec = 'w', 
                 c = 'mediumslateblue', 
                 label = label)
            ax.legend(prop = font)
            ax.tick_params(labelsize=10)
            
            ax = plt.gca()
            ax.spines['bottom'].set_linewidth(linewidth)
            ax.spines['left'].set_linewidth(linewidth)
            ax.spines['top'].set_linewidth(linewidth)
            ax.spines['right'].set_linewidth(linewidth)
            ax.set_xlabel('Iterations number', font)
            ax.set_ylabel('Crosses validation score', font)
            plt.grid(grid, linestyle= grid_linestyle)
            plt.savefig(file_name + '.jpg', dpi = dpi)