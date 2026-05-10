# encoding utf-8
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from abc import  ABCMeta, abstractmethod
from ..model import ClassificationModelfitProcessing
import  matplotlib.pyplot as plt
import seaborn as sns 

class ClassificationMetrics():
    """
    Classification metrics
    
    Args:
        True_value: list, numpy, pandas.DataFrame
            Actual target properties of the material.
        Predicted_value: list, numpy, pandas.DataFrame
            Target properties of the material predicted by the model.
        label: dict
            Category label dictionary, keys is the material target property, 
            values is the numerical property label.
    """
    def __init__(self, True_value, Predicted_value, label):
        self.True_value = True_value
        self.Predicted_value = Predicted_value
        self._label = label 

    @property
    def __multi_class(self):
        self._multi_class = []
        for i in range(len(list(self._label.values()))):
            self._multi_class = self._multi_class + list(self._label.values())[i]
        
        
        return self._multi_class

    @property
    def __multi_label(self):
        self._multi_label = list(self._label.keys())
        
        return self._multi_label

    @property
    def accuracy_number(self, normalize = False):
        """
        Accuracy number.

        Returns: int
        """
        from sklearn.metrics import accuracy_score
        self._accuracy_number = {}
        self._accuracy_number['accuracy_number'] = accuracy_score(self.True_value, self.Predicted_value, normalize = normalize)
        return self._accuracy_number

    @property
    def accuracy_score(self, normalize = True):
        """
        Accuracy score.

        Returns: float
        """
        from sklearn.metrics import accuracy_score
        self._accuracy_score = {}
        self._accuracy_score['accuracy_score'] = accuracy_score(self.True_value, self.Predicted_value, normalize = normalize)
        return self._accuracy_score

    @property
    def _is_binary(self):
        """
        Determine if binary or multi-category.

        Returns: bool
            If yes, binary classification,
            if no, multi-category.
        """
        if len(self.__multi_label) == 2:
            return True
        elif len(self.__multi_label) > 2:
            return False 

    def confusion_matrix(self, normalize = None, color = 'coral', plot = True, dpi = 600):
        """
        Confusion matrix

        Args:
            normalize: {'true', 'pred', 'None'}, default = None
                Normalizes confusion matrix over the true (rows), predicted (columns).
            color: str
                color.
            plot: bool
                Whether to plot or not.
            dpi: int, default = 600
                Image resolution.
        """
        from sklearn.metrics import confusion_matrix
        self._confusion_matrix = confusion_matrix(self.True_value, self.Predicted_value, labels = self.__multi_class, normalize = normalize)
        if plot == True:
            fig ,ax = plt.subplots(1, 1)
            ax = sns.heatmap(self._confusion_matrix, cmap = 'Blues', linewidths = 0.2, annot = True, fmt='d', annot_kws={'size':20, 'weight':'bold', 'color':'coral'})
            ax.set_xticklabels(list(self._label.keys()), {'fontsize':10, 'fontweight' : 'medium'})
            ax.set_yticklabels(list(self._label.keys()), {'fontsize':10, 'fontweight' : 'medium'})
            for label in ax.get_yticklabels():
                label.set_verticalalignment('center')
            ax.set_title('Confusion Matrix',fontsize=14, color='b',verticalalignment = 'bottom')
            ax.set_xlabel('True label',{'weight':'semibold'})
            ax.set_ylabel('Predict label',{'weight':'semibold'})
            plt.tight_layout()
            plt.savefig('Confusion_matrix.jpg', dpi = 600)
        return self._confusion_matrix

    @property
    def precision_score(self):
        """
        Precision score

        Returns: float
        """
        from sklearn.metrics import precision_score
        self._precision_score = {}
        if self._is_binary == True:
            self._precision_score['precision_binary'] = precision_score(self.True_value, self.Predicted_value, average = 'binary', labels = self.__multi_class)
        elif self._is_binary == False:
            self._precision_score['micro'] = precision_score(self.True_value, self.Predicted_value, average = 'micro', labels = self.__multi_class)
            self._precision_score['macro'] = precision_score(self.True_value, self.Predicted_value, average = 'macro', labels = self.__multi_class)
        return self._precision_score

    @property
    def recall_score(self):
        """
        Recall score

        Returns: float
        """
        from sklearn.metrics import recall_score
        self._recall_score = {}
        if self._is_binary == True:
            self._recall_score['recall_binary'] = recall_score(self.True_value, self.Predicted_value, average = 'binary', labels = self.__multi_class)
        elif self._is_binary == False:
            self._recall_score['recall_micro'] = recall_score(self.True_value, self.Predicted_value, average = 'micro', labels = self.__multi_class)
            self._recall_score['recall_macro'] = recall_score(self.True_value, self.Predicted_value, average = 'macro', labels = self.__multi_class)
        return self._recall_score

    @property
    def f1_score(self):
        """
        A measure of the accuracy of a binary classification model. It takes into account both 
        the accuracy and recall of the classification model.
    
        Returns: float
        """
        from sklearn.metrics import f1_score
        self._f1_score = {}
        if self._is_binary == True:
            self._f1_score['f1_score_binary'] = f1_score(self.True_value, self.Predicted_value, average = 'binary', labels = self.__multi_class)
        elif self._is_binary == False:
            self._f1_score['f1_score_micro'] = f1_score(self.True_value, self.Predicted_value, average = 'micro', labels = self.__multi_class)
            self._f1_score['f1_score_macro'] = f1_score(self.True_value, self.Predicted_value, average = 'macro', labels = self.__multi_class)
        return self._f1_score
    
    def roc_auc(self, plot = True):
        """
        Area Under roc Curve/Receiver Operating Characteristic

        Args:
            plot: bool
                Whether to plot or not.
        
        Returns: float
            fpr, tpr, thshold, auc
        """
        from sklearn.metrics import roc_auc_score, roc_curve, auc
        self._fpr, self._tpr, self._thesholds = roc_curve(self.True_value, self.Predicted_value,drop_intermediate = False)
        roc_auc = auc(self._fpr, self._tpr)
        if plot == True:
            font = {#'family' : 'Time New Roman',
                  'weight' : 'semibold',
                  'style' : 'normal',
                  'size' : 12
                    }
            font_1 = {#'family' : 'Time New Roman',
                  'weight' : 'semibold',
                  'style' : 'normal',
                  'size' : 10
                    }
            fig, ax = plt.subplots(1,1)
            ax.plot(self._fpr, self._tpr, label='ROC(area = %0.2f)' % (roc_auc))
            ax.plot([0, 1], [0, 1], color='navy',  linestyle='--')
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("FPR (False Positive Rate)",font)
            ax.set_ylabel("TPR (True Positive Rate)", font)
            ax.set_title("Receiver Operating Characteristic, ROC(AUC = %0.2f)"% (roc_auc), font)
            plt.legend(loc="lower right",prop = font_1)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['top'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            plt.tight_layout()
            plt.savefig('roc_auc.jpg', dpi = 600)
        self._AUC = roc_auc_score(self.True_value, self.Predicted_value)
        return self._fpr, self._tpr, self._thesholds, self._AUC

class DerandomizationClassificationMetrics(ClassificationModelfitProcessing):

    """
        Model derandomisation, as part of the model is an integrated model, there are random seed 
        parameters within it and there will be some differences in each fit, eliminating this gap 
        and taking the thought of averaging multiple fits to eliminate the random case.
        
        Args:
            fixed_model: class
                Set up the hyperparameter model.
            evaluation_index: str, list
                Evaluate the metrics, either as a string or as a list.
                Optional items:{'mae', 'mse', 'rmse', 'r2', 'msle', 'rmsle'}
            label: dict
                Category label dictionary, keys is the material target property, 
                values is the numerical property label.
            iteration_number: int
                Number of iterations
        """
    
    def __init__(self, fixed_model, evaluation_index, label, iteration_number = 20):

        super(DerandomizationClassificationMetrics, self).__init__(fixed_model)
        self.evaluation_index = evaluation_index
        self.iteration_number = iteration_number
        self.label = label
    
    @property
    def __evaluation_type(self):
        if isinstance(self.evaluation_index, str):
            self._evaluation_index = []
            self._evaluation_index.append(self.evaluation_index)
        elif isinstance(self.evaluation_index, list):
            self._evaluation_index = self.evaluation_index
        return self

    def derandomization(self, X_train, y_train, X_test, y_test):
        """
        Args: 
            X_train: numpy, list, pandas.DataFrame
                Training feature set
            y_train: numpy, list, pandas.DataFrame
                Training target set
            X_test: numpy, list, pandas.DataFrame
                Testing feature set
            y_test: numpy, list, pandas.DataFrame
                Testing target set
        
        Returns: dict
            Return averaging evaluation index
        """
        self.__evaluation_type
        output = []
        for i in range(self.iteration_number):
            one_output = []
            for j in range(len(self._evaluation_index)):
                super(DerandomizationClassificationMetrics, self).model_fit(X_train,y_train)
                y_predict_test = super(DerandomizationClassificationMetrics, self).model_predict(X_test)
                metrics = ClassificationMetrics(y_test, y_predict_test, label =self.label)
                one_output.append(getattr(metrics, self._evaluation_index[j]))
            one_output_dic = {}
            for m in range(len(one_output)):
                one_output_dic = dict(one_output_dic, **one_output[m])
            one_keys = list(one_output_dic.keys())
            one_values = list(one_output_dic.values())
            output.append(one_values)
        
        w = output[0]
        for i in range(self.iteration_number-1):
            w = [w[j] + output[i+1][j] for j in range(0, len(output[0]))]
    
        for i in range(len(w)):
            w[i] = w[i]/int(self.iteration_number)
            
        self._output = dict(zip(one_keys, w))
        return self._output