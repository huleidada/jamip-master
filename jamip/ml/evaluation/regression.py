# encoding utf-8
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import operator
from ..model import RegressionModelfitProcessing
import matplotlib.pyplot as plt
import matplotlib.colorbar as clb
from scipy.stats import norm

class RegressionMetrics():
    """
    Regression metrics

    Args:
        True_value: list, numpy, pandas.DataFrame
            Actual target properties of the material.
        Predicted_value: list, numpy, pandas.DataFrame
            Target properties of the material predicted by the model.
    """
    def __init__(self, True_value, Predicted_value):
        self.True_value = True_value
        self.Predicted_value = Predicted_value

    @property
    def mae(self):
        """
        Mean absoluate error.

        Returns: float
        """
        self._mae = np.sum(np.absolute(np.array(self.True_value) - np.array(self.Predicted_value)))/len(self.True_value)
        return self._mae

    @property
    def mse(self):
        """
        Mean square error.

        Returns: float
        """
        self._mse = np.sum((np.array(self.True_value) - np.array(self.Predicted_value))**2)/len(self.True_value)
        return self._mse

    @property
    def rmse(self):
        """
        Root mean square error.

        Returns: float
        """
        self._rmse = np.sqrt(self.mse)
        return self._rmse

    @property
    def r2(self):
        """
        Coefficient of determination.

        Returns： float
        """
        self._r2 = 1 - (np.sum((np.array(self.True_value) - np.array(self.Predicted_value))**2)/len(self.True_value))/np.var(self.True_value)
        return self._r2

    @property
    def msle(self):
        """
        Mean squared log error.

        Returns: float
        """
        self._msle = np.sum(np.log( 1 + np.array(self.True_value)) - np.log(1 + np.array(self.Predicted_value)))/len(self.True_value)
        return self._msle
    
    @property
    def rmsle(self):
        """
        Root mean squared log error.

        Returns: float
        """
        if self.msle < 0:
            raise ValueError('Mean Squared Logarithmic Error cannot be used when targets contain negative values!')
        else:
            self._rmsle = np.sqrt(self._msle)
        return self._rmsle


class DerandomizationRegessionMetrics(RegressionModelfitProcessing):
    
    def __init__(self, fixed_model, evaluation_index, iteration_number = 10):
        
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
            iteration_number: int
                Number of iterations
        """

        super(DerandomizationRegessionMetrics, self).__init__(fixed_model)

        self.evaluation_index = evaluation_index
        self.iteration_number = iteration_number

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
        for i in range(len(self._evaluation_index)): 
            one_output = []
            for j in range(self.iteration_number):
                super(DerandomizationRegessionMetrics, self).model_fit(X_train, y_train)
                y_predict_test = super(DerandomizationRegessionMetrics, self).model_predict(X_test)
                metrics = RegressionMetrics(y_test, y_predict_test)
                one_output.append(getattr(metrics, self._evaluation_index[i]))
            mean_one_output = np.mean(one_output)
            output.append(mean_one_output)
        self._output = dict(zip(self._evaluation_index, output))
        return self._output

class PlotRegression(RegressionModelfitProcessing):
    """
    Visualisation of the fit of the regression model.

    Args:
        model: str
            model name.
        fixed_model: class
            Set up the hyperparameter model.
    """
    def __init__(self, model, fixed_model):
        super(PlotRegression, self).__init__(fixed_model)
        self.model = model
    
    def plot_actuality_predict_scatter(self, target, X_train, y_train, X_test, y_test, evaluation_index, 
                                       derandomization = True, 
                                       iteration_number = 10,
                                       color = 'mediumspringgreen',
                                       size = 25,
                                       marker = 'o',
                                       linewidth = 1.5,
                                       grid = True,
                                       grid_linestyle = '--',
                                       file_name = 'Actuality_Predict',
                                       file_format = 'jpg',
                                       text_x = 2.5,
                                       text_y = 0.5,
                                       dpi = 600):
        """
        Actual vs. predicted scatter plot

        Args:
            target: str
                Predicted target value.
            X_train: numpy, list, pandas.DataFrame
                Training feature set.
            y_train: numpy, list, pandas.DataFrame
                Training target set.
            X_test: numpy, list, pandas.DataFrame
                Testing feature set.
            y_test: numpy, list, pandas.DataFrame
                Testing target set.
            evaluation_index：str, list
                Evaluate the metrics, either as a string or as a list.
                Optional items:{'mae', 'mse', 'rmse', 'r2', 'msle', 'rmsle'}.
            derandomization: bool, default = True
                Whether to derandomize or not.
            iteration_number: int, default = 10
                Number of iterations
            color: str, default = 'mediumspringgreen'
                Color.
            size: int, float, default = 25
                Scatter size.
            marker: str, default = 'o'
                Scatter markers.
            linewidth: float or int, default = 1.5
                Image edge width.
            grid: bool, default = True
                Whether the network is represented in the image.
            grid_linestyle: str, default = '--'
                Network form.
            file_name: str
                Image name.
            file_format: str,default = 'jpg'
                File format, {'svg','jpg','png'...}.
            text_x: float, int, default = 2.5
                Text annotation position x.
            text_y : float, int, default = 0.5
                Text annotation position y.
            dpi: int, default = 600
                Image resolution.
        
        Returns:
            figure
        """
        super(PlotRegression, self).model_fit(X_train, y_train)
        y_predict_test = super(PlotRegression, self).model_predict(X_test)
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        ax = plt.subplot(111)
        font = {
                  'weight' : 'semibold',
                  'style' : 'normal',
                  'size' : 10
                    }
        plt.scatter(
            y_test, y_predict_test, 
            s = size, 
            c = color, 
            edgecolors = 'k',
            marker = marker,
            norm = 0.8, 
            linewidths = None, 
            alpha = 1,
            label = target
            )
        MIN = min(y_test)
        MAX = max(y_predict_test)
        x = np.arange(MIN, MAX, 0.01)
        ax.plot(x,x, 'r--', lw = 1, zorder = 0, color = 'black')
        ax.legend(prop = font)
        ax.grid(grid, linestyle = grid_linestyle)
        ax.set_xlabel('Actual ' + target, fontsize = 10, fontweight = 'semibold')
        ax.set_ylabel(self.model + 'predictive ' + target, fontsize = 10, fontweight = 'semibold')
        ax.set_ylabel(self.model + ' predictive '  + target, fontsize = 10, fontweight = 'semibold')
        ax.spines['top'].set_linewidth(linewidth)
        ax.spines['right'].set_linewidth(linewidth)
        ax.spines['left'].set_linewidth(linewidth)
        ax.spines['bottom'].set_linewidth(linewidth)
        
        if isinstance(evaluation_index,str):
            metrics_index = []
            metrics_index.append(evaluation_index)
        elif isinstance(evaluation_index,list):
            metrics_index = evaluation_index
        
        output = []
        if derandomization:  
            if isinstance(iteration_number, int):
                output = list(DerandomizationRegessionMetrics(self.fixed_model,metrics_index, iteration_number).derandomization(X_train,y_train,X_test,y_test).values())
            else:
                ValueError("Iteration number must be 'int'!")
        else:
            for i in range(len(metrics_index)):
                output.append(
                    getattr(RegressionMetrics(y_test, y_predict_test),
                            metrics_index[i])
                )
        z = ' '
        for i in range(len(output)):
            r = metrics_index[i] + '= ' +str('%.4f' % output[i] + '\n')
            z = z + r
        ax.text(text_x, text_y, z, fontsize=14, style='oblique', weight = 'roman')
        plt.tight_layout()
        plt.savefig(file_name + '.' + file_format, dpi = dpi)
    
    def plot_data_distribution(self, X_train, y_train, X_test, y_test, Gaussian = True,
                               color_up = 'mediumspringgreen',
                               color_down = 'mediumorchid',
                               bins = 40,
                               linewidth = 1.5,
                               grid = True,
                               grid_linestyle = '--',
                               file_name = 'Data_distribution_comparison',
                               file_format = 'jpg',
                               text_x = 0.75,
                               text_y = 0.75,
                               dpi = 600):
        """
        Actual vs. predicted bar distribution

        Args:
             X_train: numpy, list, pandas.DataFrame
                Training feature set.
            y_train: numpy, list, pandas.DataFrame
                Training target set.
            X_test: numpy, list, pandas.DataFrame
                Testing feature set.
            y_test: numpy, list, pandas.DataFrame
                Testing target set.
            Gaussian: bool, default = True
                Whether to draw a Gaussian curve.
            bins: int, default = 40
                Number of columns.
            linewidth: float or int, default = 1.5
                Image edge width.
            grid: bool, default = True
                Whether the network is represented in the image.
            grid_linestyle: str, default = '--'
                Network form.
            file_name: str
                Image name.
            file_format: str,default = 'jpg'
                File format, {'svg','jpg','png'...}.
            text_x: float, int, default = 0.75
                Text annotation position x.
            text_y : float, int, default = 0.75
                Text annotation position y.
            dpi: int, default = 600
                Image resolution.

        Returns:
            figure
        """
        super(PlotRegression, self).model_fit(X_train, y_train)
        predictive_value = super(PlotRegression, self).model_predict(X_test)
        X_MIN = min(min(y_test), min(predictive_value))
        X_MAX = max(max(y_test),max(predictive_value))
        font = {
                'weight' : 'semibold',
                'style' : 'normal',
                'size' : 10
                }
        fig,ax = plt.subplots(nrows = 2, ncols = 1)
        ax1 = plt.subplot(211)
        plt.title('Data distribution comparison', font)
        n ,bins ,patches = ax1.hist(x = y_test,  
                                    bins = bins, 
                                    color = color_up, 
                                    edgecolor = 'k',  
                                    label = 'True value distribution')
        ax1.legend(prop = font)
        ax1.grid(grid, linestyle = grid_linestyle)
        ax1.set_xlim([X_MIN, X_MAX])                                             
        y_max = max(n)
        ax1.set_ylim([0, y_max * 1.1])
        ax1.set_ylabel('Counts', fontsize = 10, fontweight = 'semibold')
        ax1.spines['top'].set_linewidth(linewidth)
        ax1.spines['right'].set_linewidth(linewidth)
        ax1.spines['left'].set_linewidth(linewidth)
        ax1.spines['bottom'].set_linewidth(linewidth)
        if Gaussian == True:
            mu = np.mean(y_test)
            sigma= np.std(y_test)
            y = norm.pdf(bins, mu, sigma)
            ax11 = ax1.twinx()
            ax11.plot(bins, y, 'r--')
            ax11.set_ylim([0,1.1*max(y)])
            ax11.set_ylabel('Probability density', 
                            fontsize = 10,
                            fontweight = 'semibold')
            ax1.annotate('$\mu = $' + str('%.2f' % mu) + ' $\sigma = $' + str('%.2f' % sigma), 
                         xy = [mu*text_x, y_max*text_y])
        
        ax2 = plt.subplot(212)
        n_predict, bins_predict, patches_predict = ax2.hist(x = predictive_value, 
                                                            bins = bins, 
                                                            color = color_down, 
                                                            edgecolor = 'k', 
                                                            label = 'Predictive value distribution')
        ax2.legend(prop = font)
        ax2.grid(True, linestyle = '--')
        ax2.set_xlim([X_MIN, X_MAX])
        y_predict_max = max(n_predict)
        ax2.set_ylim([0, y_predict_max*1.1])
        ax2.set_xlabel('Actual/Predictive data distribution', fontsize = 10, fontweight = 'semibold')
        ax2.set_ylabel('Counts', fontsize = 10, fontweight = 'semibold')
        ax2.spines['top'].set_linewidth(linewidth)
        ax2.spines['right'].set_linewidth(linewidth)
        ax2.spines['left'].set_linewidth(linewidth)
        ax2.spines['bottom'].set_linewidth(linewidth)
        if Gaussian == True:
            mu_predict = np.mean(predictive_value)
            sigma_predict = np.std(predictive_value)
            y_predict = norm.pdf(bins_predict, mu_predict, sigma_predict)
            ax22 = ax2.twinx()
            ax22.plot(bins_predict, y_predict, 'r--')
            ax22.set_ylim([0,1.1*max(y_predict)])
            ax22.set_ylabel('Probability density', fontsize = 10, fontweight = 'semibold')
            ax2.annotate('$\mu = $' + str('%.2f' % mu_predict) + ' $\sigma = $' + str('%.2f' % sigma_predict), xy = [mu_predict*text_x, y_predict_max*text_y])
        plt.tight_layout()
        plt.savefig(file_name + '.' + file_format, dpi = dpi)