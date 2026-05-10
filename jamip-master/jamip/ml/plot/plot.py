# encoding utf-8
#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colorbar as clb
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ..evaluation import RegressionMetrics, DerandomizationRegessionMetrics
from ..evaluation import ClassificationMetrics, DerandomizationClassificationMetrics

class Plot(object):
    """
    Args:
        df: pandas.DataFrame
            Dataframe containing all features and target.
        linewidth: int, float, default = 1.5
            Image edge width.
        dpi: int, default = 600
            Image resolution.
        grid: bool, default = True
            Whether the network is represented in the image.
        grid_linestyle: str, default = '--'
            Network form.   
    """
    def __init__(self, df, 
                       linewidth = 1.5,
                       dpi = 600,
                       grid = True,
                       grid_linestyle = '--',
                       ):

        self.df = df
        self.linewidth = linewidth
        self.dpi = dpi
        self.grid = grid
        self.grid_linestyle = grid_linestyle

    def feature_target_relationship(self, 
                                    feature, 
                                    target, 
                                    file_name,
                                    file_format = 'jpg',
                                    size = 25,
                                    color = 'mediumorchid', 
                                    marker = 'o',
                                    alpha = 0.4,
                                    fontsize = 10,
                                    fontweight = 'semibold',
                                    edgecolors = None,
                                    linewidths = 0.5):
        """
        single feature and target relationship plot

        Args:
            feature: str
                Selecting the features to be normalised in the dataframe.
            target: str
                Name of the target property to be fitted in the dataframe.
            file_name: str
                Image name.
            file_format: str,default = 'jpg'
                File format, {'svg','jpg','png'...}.
            size: int, float, default = 25
                Scatter size.
            color: str, default = 'mediumspringgreen'
                Color.
            marker: str, default = 'o'
                Scatter markers.
            alpha: float, default = 0.4
                Alpha.
            fontsize: int, float, default = 10
                Font size.
            fontweight: str, default = 'semibold'
                Font thickness.
            edgecolors: str, None, default = None
                Edge color.
            linewidths: float, int, default = 0,5
                Scatter edge width.
        """
        feature_df = self.df[feature]
        target_df = self.df[target]
        fig,ax = plt.subplots(nrows = 1, ncols = 1)
        ax = plt.subplot(111)
        ax.scatter(feature_df, target_df, 
                   s = size, 
                   c = color, 
                   marker = marker,  
                   linewidths = linewidths, 
                   alpha = alpha,
                   edgecolors = edgecolors)

        ax.grid(self.grid, linestyle = self.grid_linestyle)
        ax.set_xlabel(feature, fontsize = fontsize, fontweight = fontweight)
        ax.set_ylabel('DFT '  + target  , fontsize = fontsize, fontweight = fontweight)
        ax.spines['bottom'].set_linewidth(self.linewidth)
        ax.spines['left'].set_linewidth(self.linewidth)
        ax.spines['top'].set_linewidth(self.linewidth)
        ax.spines['right'].set_linewidth(self.linewidth)
        plt.savefig(file_name + '.' + file_format, dpi = self.dpi)
    
    def feature_pair_target_relationship(self, 
                                         features, 
                                         target,
                                         file_name,
                                         file_format = 'jpg',
                                         size = 25,
                                         cmap = 'coolwarm',
                                         marker = 'o',
                                         alpha = 0.9,
                                         fontsize = 10,
                                         fontweight = 'semibold',
                                         edgecolors = None,
                                         linewidths = 1
                                         ):
        """
        Double features and target relationship plot

        Args:
            feature: str
                Selecting the features to be normalised in the dataframe.
            target: str
                Name of the target property to be fitted in the dataframe.
            filen_name: str
                Image name.
            file_format: str,default = 'jpg'
                File format, {'svg','jpg','png'...}.
            size: int, float, default = 25
                Scatter size.
            color: str, default = 'coolwarm'
                Color bar.
            marker: str, default = 'o'
                Scatter markers.
            alpha: float, default = 0.9
                Alpha.
            fontsize: int, float, default = 10
                Font size.
            fontweight: str, default = 'semibold'
                Font thickness.
            edgecolors: str, None, default = 'semibold'
                Edge color.
            linewidths: float, int, default = 0,5
                Scatter edge width.
        """
        features_df = self.df[features]
        target_df = self.df[target]
        fig,ax = plt.subplots()
        ax = plt.subplot(111)
        sc = ax.scatter(features_df[features[0]], features_df[features[1]], 
                        marker=marker, 
                        c = target_df,
                        s = size, 
                        cmap = cmap, 
                        alpha = alpha, 
                        linewidths = linewidths,
                        edgecolors = edgecolors)

        plt.colorbar(sc)
        ax.set_xlabel(features[0], fontsize = 10, fontweight = 'semibold')
        ax.set_ylabel(features[1], fontsize = 10, fontweight = 'semibold')
        ax.spines['top'].set_linewidth(self.linewidth)
        ax.spines['right'].set_linewidth(self.linewidth)
        ax.spines['left'].set_linewidth(self.linewidth)
        ax.spines['bottom'].set_linewidth(self.linewidth)
        plt.grid(self.grid, linestyle = self.grid_linestyle)
        plt.savefig(file_name + '.' + file_format, dpi = self.dpi)
    
    def plot_violin(self, 
                    features,
                    target,
                    file_name,
                    file_format = 'jpg',
                    color = 'mediumorchid',
                    linewidths = 1,
                    scale = 'count',
                    inner = 'box',
                    saturation = 1,
                    fontsize = 10,
                    fontweight = 'semibold',
                    ):
        """
        Violin Plot

        Args:
            features: str, list
                Selecting the features to be normalised in the dataframe.
            target: str
                Name of the target property to be fitted in the dataframe.
            filen_name: str
                Image name.
            file_format: str,default = 'jpg'
                File format, {'svg','jpg','png'...}.
            color: str, default = 'mediumorchid'
                Color.
            linewidths: float, int, default = 1
                Violin plot edge width.
            scale: {'area','count', 'width'}, default = 'count'
                Used to scale the width of each violin.
            inner: {'box'，'quartile'，'point'，'stick'，None}, defaulr = 'box'
                Indicates the data points inside the violin.
            saturation: float, default = 1
                Saturation.
            fontsize: int, float, default = 10
                Font size.
            fontweight: str, default = 'semibold'
                Font thickness.
        """
        if isinstance(features, str):
            data = self.df[[features]]
        elif isinstance(features, list):
            data = self.df[features]
        sns.violinplot(data = data,
                       color = color,
                       linewidth = linewidths,
                       scale = scale,
                       inner = inner,
                       saturation = saturation)
        plt.ylabel(target, fontsize = fontsize , fontweight = fontweight)
        ax = plt.gca()
        ax.spines['top'].set_linewidth(self.linewidth)
        ax.spines['right'].set_linewidth(self.linewidth)
        ax.spines['left'].set_linewidth(self.linewidth)
        ax.spines['bottom'].set_linewidth(self.linewidth)
        plt.savefig(file_name + '.' + file_format, dpi = self.dpi)
    
    def plot_box(self,
                 features,
                 file_name,
                 file_format = 'jpg',
                 meanprops = {'marker':'*','color':'black','markerfacecolor':'palegreen', 'markersize':10},
                 medianprops = {'linestyle':'--', 'color':'red'},
                 sym = 'o',
                 filterprops = None,
                 patch_artist = False,
                 y_label = None,
                 fontsize = 10,
                 fontweight = 'semibold',
                 ):
        """
        Box Plot

        Args:
            features: str, list
                Selecting the features to be normalised in the dataframe.
            filen_name: str
                Image name.
            file_format: str,default = 'jpg'
                File format, {'svg','jpg','png'...}.
            meanprops: None, dict 
                Attribute of the mean point
            medianprops: None, dict
                Properties of the median
            sym: str, default = 'o' 
                Outlier shape   
            filterprops: None, dict
                Properties of outliers
            patch_artist: bool, defaulr = False
                Whether or not the box is filled with the color
            y_label: str, default = None
                y-axis label
            fontsize: int, float, default = 10
                Font size.
            fontweight: str, default = 'semibold'
                Font thickness.
        """
        if isinstance(features, str):
            data = self.df[[features]]
        elif isinstance(features, list):
            data = self.df[features]
        data_list = []
        for i in list(data.columns.values):
            data_list.append(data[i].to_list())
        plt.boxplot(
                    data_list,
                    showmeans = True,
                    meanprops = meanprops,
                    medianprops = medianprops,
                    sym = sym,
                    flierprops = filterprops,
                    patch_artist = patch_artist,
                    labels = list(data.columns.values)
            )
        if isinstance(y_label,str): 
            plt.ylabel(y_label, fontsize = fontsize , fontweight = fontweight)
        #plt.xlabel(ylabel, fontsize = fontsize , fontweight = fontweight)
        ax = plt.gca()
        ax.spines['top'].set_linewidth(self.linewidth)
        ax.spines['right'].set_linewidth(self.linewidth)
        ax.spines['left'].set_linewidth(self.linewidth)
        ax.spines['bottom'].set_linewidth(self.linewidth)
        plt.grid(self.grid,  linestyle = self.grid_linestyle)
        plt.savefig(file_name + '.' + file_format , dpi = self.dpi)