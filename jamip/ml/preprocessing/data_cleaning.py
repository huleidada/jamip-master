# encoding utf-8
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import math
from collections import Counter
from scipy import stats

def BoxPlotAnalysis(df, col, 
                    lower_quantile_limit = 0.25, 
                    upper_quantile_limit = 0.75):
    """
    Args:
        df(pandas.DataFrame): Dataframe containing all features. 
        cols: Feature names that need to be BoxPlotAnalysis.
        lower_quantile_limit/upper_quantile_limit(float): Box plot analyse thresholds.
    Returns:
        lower_limit/upper_limit(int or float): Column-specific upper and lower values.
        outer_samples(pandas.DataFrame): Outlier data in dataframe format.
    """
    if str(df[col].dtype) == 'object' or str(df[col].dtype) == 'bool':
        raise ValueError("Box plot analysis is impossible for both category values and Bool values!")
    else:    
        q1 = df[col].quantile(lower_quantile_limit, interpolation = 'linear')
        q3 = df[col].quantile(upper_quantile_limit, interpolation = 'linear')
        iqr = q3-q1
        lower_limit = q1 - 1.5*iqr
        upper_limit = q3 + 1.5*iqr
    
    l = (df[col] > lower_limit)
    u = (df[col] < upper_limit)
    intersection_bool = (l==u)
    outer_samples = df[~intersection_bool]
    return lower_limit, upper_limit, outer_samples


def ThreeSigmaAnalysis(df, col):
    """
    Args:
        df(pandas.DataFrame): Dataframe containing all features. 
        cols: Feature names that need to be ThreeSigmaAnalysis.
    Returns:
        u/std(int or float): Means and standard deviations of normal distributions
        outer_samples(pandas.DataFrame): Outlier data in dataframe format.
    """
    if str(df[col].dtype) == 'object' or str(df[col].dtype) == 'bool':
        raise ValueError("3σ analysis is impossible for both category values and Bool values!")
    else:
        u = df[col].mean()
        std = df[col].std()
        p_value = stats.kstest(df[col], 'norm', (u, std))[1]
        if p_value <= 0.05:
            outer_samples = df[np.abs(df[col] - u) > 3*std]
        else:
            import warnings
            warnings.warn('The data does not conform to a normal distribution'
                          'and is not suitable for 3σ analysis!')

    return u, std, outer_samples


class DataCleaning(object):
    
    """
    Convert the initial dataframe into ML-ready dataframe.

    The process is to first pre-check the null position of the overall 
    dataframe, then drop the target samples with nan values, then process 
    each feature position of the samples with nan values in turn, and finally 
    check the distribution of the features and process the features with 
    abnormal distribution.

    Args:
        dataset_df(pandas.DataFrame): Dataframe containing all features 
            and targets.
        target(str): Name of the target property to be fitted in the dataframe.
    """
    def __init__(
        self,
        dataset_df,
        target,
    ):
    
        self.__df = dataset_df
        self._df_original = dataset_df
        if target in list(self.__df.columns.values):
            self.target = target
        else:
            raise ValueError(
        "The {} is not in the dataset !".format(target)
            )
        self.cols = list(self.__df.columns.values)

    @property
    def df_original(self):
        return self._df_original

    def number_object_cols(self):
        """
        Identifying numeric and str columns in a dataframe.

        Returns: self
        """
        self._number_fea_cols = []
        self._object_fea_cols = []

        for i in self.__df.columns.values:
            try:
                if self.__df[i].dtype == bool:
                    # True -> 1, False -> 0
                    self.__df[i] = self.__df[i].astype(int)
                else:
                    self.__df[i] = pd.to_numeric(self.__df[i])
                if i != self.target:
                    self._number_fea_cols.append(i)
            except(TypeError, ValueError):
                if i != self.target:
                    self._object_fea_cols.append(i)
        return self
    
    @property
    def number_feature(self):
        """
        Get numeric columns.
        
        Returns (list)
        """
        return self._number_fea_cols
    
    @property
    def object_feature(self):
        """
        Get str columns.
        
        Returns (list)
        """
        return self._object_fea_cols

    def precheck_x(self,x):
        """
        Check whether a column in a dataframe has null values.

        Args: 
            x(str): A feature name in dataframe.columns

        Returns 
            (Pandas.Series) Columns with null values

        """
        self._check_x = self.__df[self.__df[x].isnull()][x]
        return self._check_x

    def precheck_dataframe(self) -> dict:
        """
        Check whether overall dataframe has null values.

        Returns
            (dict): The keys are name of the column with null values.
                The values are number of rows with null values.
        """
        nan_cols_df = self.__df.isnull().any()
        self.nan_cols_dict = {}
        for i in list(self.__df.columns.values):
            if nan_cols_df[i] == True:        
                self.nan_cols_dict[i] = list(self.precheck_x(i).index)
            else:
                pass

        return self.nan_cols_dict
    
    def drop_nan_target(self):
        """
        Directly delete the target sample with a null value.

        Returns: self
        """
        if self.target in self.__df.columns:
            cleaned_df = self.__df.dropna(axis = 0, how = 'any', subset = [self.target])
            self.deled_samples = self.__df[~self.__df.index.isin(cleaned_df.index)]
            self.__df = cleaned_df

        elif self.target not in self.__df.columns:
            import warnings
            warnings.warn('This target has not been in the dataset!')
        
        return self
    
    def process_nan_x(self, x, nan_method, fill_value = False):
        """
        Process a feature in the dataframe. We offer a number of different ways
        to handle this, including delete, top and bottom fill, average fill 
        and specific value fill, This step is best performed after the target 
        nulls have been processed.
        If you need a different fill method for each column, select this method.

        Args:
            x(str): a feature name in the dataframe.
            nan_method(str): Setting how to handle the sample with nulls.Optional 
                objects:{'ignore', 'del', 'fill', 'mean'}, 'ignore' to ignore nan value, 
                or 'fill' to fill the upper and lower bits, 'mean' to fill the average value.
            fill_value(bool, int or float or str): If False, indicates that no specific value 
                filling method is selected. If True,  will be reminded to fill a specific value. 
                If int or float, will fill the null value with a specific value.
            
            Returns: self
        """
        self.number_object_cols()
        #self.drop_nan_target()
        
        if nan_method == 'ignore':
            print(
                "Nan values have been ignored!"
            )
            pass

        elif nan_method == 'del':
            cleaned_df = self.__df.dropna(axis = 0, how = 'any', subset = [x])
            self.deled_samples= pd.concat(
                (self.deled_samples, self.__df[~self.__df.index.isin(cleaned_df.index)]), 
                 axis = 0,
            )
            self.__df = cleaned_df
        
        elif nan_method == 'fill' and fill_value == False and isinstance(fill_value, bool):
            cleaned_df = pd.DataFrame(self.__df[x])
            processed_df = pd.DataFrame(self.__df.drop(columns = [x]))
            cleaned_df = cleaned_df.fillna(method = 'ffill')
            cleaned_df = cleaned_df.fillna(method = 'bfill')
            self.__df = pd.concat(
                (processed_df, cleaned_df),
                axis = 1,
            )
            self.__df = self.__df[self.cols]
            
        elif nan_method == 'mean':
            if str(self.__df[x].dtype) != 'object':
                cleaned_df = pd.DataFrame(self.__df[x])
                processed_df = pd.DataFrame(self.__df.drop(columns = [x]))
                cleaned_df = cleaned_df.fillna(value = cleaned_df.mean())
                self.__df = pd.concat(
                    (processed_df, cleaned_df),
                    axis = 1,
                )
                self.__df = self.__df[self.cols]
            elif str(self.__df[x].dtype) == 'object':
                raise ValueError("Object feature can't be filled processed!")
        
        elif nan_method == 'fill' and (isinstance(fill_value, int) 
                                       or isinstance(fill_value, float) 
                                       or isinstance(fill_value, str)):
            if fill_value == True:
                raise TypeError("Please input a value or object instead of 'True'!")
            else:
                cleaned_df = pd.DataFrame(self.__df[x])
                processed_df = pd.DataFrame(self.__df.drop(columns = [x]))
                cleaned_df = cleaned_df.fillna(value = fill_value)
                self.__df = pd.concat(
                    (processed_df, cleaned_df),
                    axis = 1,
                )
                self.__df = self.__df[self.cols]
        
        return self

    def process_nan_datafrme(self, nan_method, fill_value = False):
        """
        Process all the null features of the dataframe.We offer a number of 
        different ways to handle this, including delete, top and bottom fill, 
        average fill and specific value fill,This step is best performed after
        the target nulls have been processed.

        Args:
            nan_method(str): Setting how to handle the sample with nulls.Optional 
                objects:{'ignore', 'fill', 'mean'}, 'ignore' to ignore nan value, or 'fill'
                to fill the upper and lower bits, 'mean' to fill the average value.
            fill_value(bool, int or float or str): If False, indicates that no specific value filling 
                method is selected. If True,  will be reminded to fill a specific value. 
                If int or float, will fill the null value with a specific value.

        Returns: self
        """
        self.number_object_cols()
        #self.drop_nan_target()

        if nan_method == 'ignore':
            print (
                "Nan values has been ignored !"
            )
            pass 

        elif nan_method == 'del':
            cleaned_df = self.__df.dropna(
                axis = 0, how = 'any'
                )
            self.deled_samples= pd.concat(
                (self.deled_samples, self.__df[~self.__df.index.isin(cleaned_df.index)]), 
                 axis = 0,
            )
            self.__df = cleaned_df
        
        elif nan_method == 'fill' and fill_value == False and isinstance(fill_value, bool):
            
            self.__df = self.__df.fillna(method = 'ffill')
            self.__df = self.__df.fillna(method = 'bfill')
        
        elif nan_method == 'mean':
            #numberic nan
            df_mean = self.__df[[ncol for ncol in self.__df.columns if ncol in self.number_feature]]
            df_mean = df_mean.fillna(value = df_mean.mean())
            self.__df[df_mean.columns] = df_mean

            #object nan
            self.__df = self.__df.fillna(method = 'ffill')
            self.__df = self.__df.fillna(method = 'bfill')
        
        elif nan_method == 'fill' and (isinstance(fill_value, int) 
                                       or isinstance(fill_value, float) 
                                       or isinstance(fill_value, str)):
            if fill_value == True:
                raise TypeError("Please input a value or object instead of 'True' !")
            else:
                self.__df = self.__df.fillna(value = fill_value)
        return self
    
    def process_outlier_dataframe(self, x,  
                                  outlier_method,
                                  process_outlier_method, 
                                  lower_quantile_limit = 0.25, 
                                  upper_quantile_limit = 0.75,
                                  fill_value = False):
        """
        Analyses and processes the distribution of a particular column of a dataframe.
        Two analysis methods are provided: Box plot and 3σ principle analysis. 
        For outlier data provide ignore/del/mean/fill_value ways to handle.

        Args:
            x(str): a feature name in the dataframe.
            outlier_method(str): Set how to analyse data outlier methods, Optional 
                objects:{'box', 'three_sigam'}, 'box' need to set upper and lower
                quantile limit, while 'three_sigam' needs to determine if the data 
                meets the distribution.
            precess_outlier_method(str): Setting how to handle the outlier samples. 
                Optional objects:{'ignore', 'del', 'fill', 'mean'}, 'ignore' to ignore 
                nan value, or 'fill'to fill specific values, 'mean' to fill the average value.
            lower_quantile_limit/upper_quantile_limit(float): Box plot analyse thresholds.
            fill_value(bool, int or float or str): If False, indicates that no specific value filling 
                method is selected. If True,  will be reminded to fill a specific value. 
                If int or float, will fill the null value with a specific value.
        
        Returns: self
        """
        self.number_object_cols()
        #self.drop_nan_target()


        if outlier_method == 'box':
            lower_limit, upper_limit ,outlier_samples = BoxPlotAnalysis(self.__df, x, 
                                                                        lower_quantile_limit = lower_quantile_limit, 
                                                                        upper_quantile_limit = upper_quantile_limit)
            print("Lower limit:{} \nUpper limit:{}".format(lower_limit, upper_limit))

        elif outlier_method == 'three_sigam':
            u, std, outlier_samples = ThreeSigmaAnalysis(self.__df, x)
            print("Mean value:{} \nStandard deviation:{}".format(u, std))
        else:
            raise ValueError(
                "No {} method of processing outliers".format(outlier_method)
            )

        if process_outlier_method == 'ignore':
            print (
                "Dataset remains unchanged!"
            )

        elif process_outlier_method == 'del':
            cleaned_df = self.__df[~self.__df.index.isin(outlier_samples.index)]
            
            self.__df = cleaned_df

        elif process_outlier_method == 'mean':
            cleaned_df = self.__df[~self.__df.index.isin(outlier_samples.index)]
            processed_outlier_df = pd.DataFrame(columns = [x], 
                                                index=list(outlier_samples.index.values)).fillna(value = cleaned_df[x].mean())
            cleaned_outlier_df = pd.DataFrame(outlier_samples.drop(columns = [x]))
            outlier_df = pd.concat(
                (cleaned_outlier_df, processed_outlier_df),
                axis = 1
            )
            outlier_df = outlier_df[self.cols]
            self.__df = cleaned_df.append(outlier_df).sort_index(ascending=True)
        
        elif process_outlier_method == 'fill' and (fill_value != False or fill_value == 0):
            if fill_value == True:
                raise TypeError("Please input a value or object instead of 'True' !")
            
            else:
                cleaned_df = self.__df[~self.__df.index.isin(outlier_samples.index)]
                processed_outlier_df = pd.DataFrame(columns = [x], 
                                                    index=list(outlier_samples.index.values)).fillna(value = fill_value)

                cleaned_outlier_df = pd.DataFrame(outlier_samples.drop(columns = [x]))
                outlier_df = pd.concat(
                    (cleaned_outlier_df, processed_outlier_df),
                    axis = 1
                )
                outlier_df = outlier_df[self.cols]
            self.__df = cleaned_df.append(outlier_df).sort_index(ascending=True)
        return self

    def transform_dataframe(self):
        """
        Converting the initial dataframe to a dataframe that has been data cleaned.

        Returns: (Pandas.DataFrame)
            The dataframe that has been data cleaned.
        """
        if (list(self.__df.columns.values) != list(self.df_original.columns.values)) or (self.__df.shape != self.df_original.shape):
            return self.__df
        elif (list(self.__df.columns.values) == list(self.df_original.columns.values)) and (self.__df.shape == self.df_original.shape):
            judgement_matrix = (self.__df == self.df_original)
            if False in (judgement_matrix.values):
                return self.__df
            else:
                import warnings
                warnings.warn('No changes have been made to the dataset!')
        else:
            import warnings
            warnings.warn('No changes have been made to the dataset!')