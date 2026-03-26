# encoding utf-8
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import math
from collections import Counter
from scipy import stats

def BinaryEncode(df, col):
    """
    Args:
        df(pandas.DataFrame): Dataframe containing all features. 
        cols: Feature names that need to be binary transcoded.
    Returns:
        (pandas.DataFrame) The dataframe that has completed binary transcoding.
    """
    cate_dict = Counter(df[col])
    
    binary_len = int(math.log2(len(cate_dict))) + 1

    for n, k in enumerate(cate_dict.keys()):
        # bin(6) --> '0b110'
        r = list(bin(n))[2:]
        r = ['0'] * (binary_len - len(r)) + r
        cate_dict[k] = "".join(r)

    col_df = df[col].map(cate_dict)
    new_df = pd.DataFrame()
    for i in range(binary_len):
        new_df[f'{col}_{i}'] = col_df.map(lambda x: list(x)[i])
    
    return pd.concat([df, new_df], axis = 1).drop(columns = [col])

class CategoryCoding(object):
    """
    Converting a dataframe with category-based features to a dataframe 
    with all numeric features.

    Args:
        dataset_df(pandas.DataFrame): Dataframe containing all features 
            and targets
        target(str): Name of the target property to be fitted in the dataframe
    """
    def __init__(self, dataset_df, target,):
        self.__df = dataset_df
        self._df_original = dataset_df
        self.cols = list(self.__df.columns.values)
        if target in list(self.__df.columns.values):
            self.target = target
        else:
            raise ValueError(
        "The {} is not in the dataset !".format(target)
            )
    
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
    
    def category_coding_x(self, x, encode_method):
        """
        The categorical feature coding for a particular feature in a dataframe 
        is handled in three different ways, including label coding, one-hot coding 
        and binary coding.
        If you need a different encoding method for each column, select this method.

        Args:
            x(str): a feature name in the dataframe.
            encode_method(str): Setting how to categorical coding. Optional object:{'label','one-hot','binary'}.

        Returns: self
        """

        self.number_object_cols()

        number_df = pd.DataFrame(self.__df[self.number_feature])
        object_df = pd.DataFrame(self.__df[self.object_feature])
        object_x_df = pd.DataFrame(self.__df[x])
        target_df = pd.DataFrame(self.__df[self.target])

        if encode_method == 'one-hot':
            print (
                "One-hot encoding is used for columns {}".format(
                        object_x_df.columns.tolist()
                        )
                    )
            object_x_df = pd.get_dummies(object_df).apply(pd.to_numeric)
        elif encode_method == 'label':
            print (
                "Label encoding is used for columns {}".format(
                object_x_df.columns.tolist()
                    )
                )
            tmp = pd.DataFrame()
            tmp[x + '_label'] = LabelEncoder().fit_transform(object_x_df)
            object_x_df = tmp
        elif encode_method == 'binary':
            print(
                "Binary encoding is used for columns {}".format(
                object_x_df.columns.tolist()
                    )
                )
            object_x_df = BinaryEncode(object_x_df,x)
        
        self.__df = pd.concat(
            [number_df, object_df, object_x_df, target_df],
            axis = 1,
            ).drop(columns = [x])

        return self
    
    def category_coding_dataframe(self, encode_method):
        """
        The categorical feature coding for all features in a dataframe is 
        handled in three different ways, including label coding, one-hot coding 
        and binary coding.

        Args:
            encode_method(str): Setting how to categorical coding. Optional object:{'label','one-hot','binary'}.

        Returns: self
        """
        self.number_object_cols()

        number_df = pd.DataFrame(self.__df[self.number_feature])
        object_df = pd.DataFrame(self.__df[self.object_feature])
        target_df = pd.DataFrame(self.__df[self.target])
        
        if object_df.empty != True:
            if encode_method == 'one-hot':
                print (
                        "One-hot encoding is used for columns {}".format(
                        object_df.columns.tolist()
                        )
                    )
                object_df = pd.get_dummies(object_df).apply(pd.to_numeric)
            elif encode_method == 'label':
                print (
                    "Label encoding is used for columns {}".format(
                        object_df.columns.tolist()
                    )
                )
                for i in object_df.columns:
                    object_df[i] = LabelEncoder().fit_transform(object_df[i])
            
            elif encode_method == 'binary':
                print (
                    "Binary encoding is used for columns {}".format(
                       object_df.columns.tolist() 
                    )
                )
                for i in object_df.columns:
                    object_df = BinaryEncode(object_df,i)
            self.__df =  pd.concat(
                [number_df, object_df, target_df], 
                axis =1)
            
            return self
        else:
            self.__df =  pd.concat(
                [number_df, target_df],
                axis =1)
            
            return self

    def transform_dataframe(self):
        """
        Converting the initial dataframe to a dataframe that has been categorical coding.

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
