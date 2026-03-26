import pandas as pd
import numpy as np

Tree_base = {
    "n_estimators" : np.arange(10,1000,5),
    "max_features" : ['auto', 'sqrt', 'log2'],
    "max_depth" : np.arange(3,50),
    "min_samples_split" : np.arange(2,21),
    "min_samples_leaf" : np.arange(1,21),
}

REGRESSION_HYPER = {
    "LinearRegression" : {
        "normalize" : [True, False]
    },
    "BayesianRidge" : {
        "n_iter": np.arange(10,500,5),
        #"tol" : [],
        #"alpha_1" : [],
        #"alpha_2" : [],
        #"lambda_1" : [],
        #"lambda_2" : [],
         "normalize" : [True, False]
    },
    "Lasoo" : {
        "normalize" : [True, False],
        "max_iter" : np.arange(1,1000,2),
        "tol" :  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    "ridge_regression" : {
        "solver" : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        "max_iter" : np.arange(1,1000,2),
        "tol" :  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    "KNeighborsRegressor" : {
        "n_neighbors": range(1, 1000, 2),
        "algorithm" : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    },

    "RadiusNeighborsRegressor" : {
        "weights": ["uniform", "distance"],
        "algorithm" : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        "p": [1, 2]
    },

    "SVR" : {
        "kernel" : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        "epsilon": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    } ,
    "NuSVR" : {
        "nu" : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "kernel" : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    },

    "LinearSVR" : {
        "loss" : ["epsilon_insensitive", "squared_epsilon_insensitive"],
        "dual": [True, False],
        "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        "epsilon": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    },

    "DecisionTreeRegressor" : {
        "max_depth": Tree_base["max_depths"],
        "min_samples_split": Tree_base["min_samples_split"],
        "min_samples_leaf": Tree_base["min_samples_leaf"],
        "max_features" : Tree_base['max_features']
    },
    "ExtraTreeRegressor" : {
        "max_depth": Tree_base["max_depths"],
        "min_samples_split": Tree_base["min_samples_split"],
        "min_samples_leaf": Tree_base["min_samples_leaf"],
        "max_features" : Tree_base['max_features']
    },
    "RandomForestRegressor" : {
        "n_estimators" : Tree_base["n_estimators"],
        "max_depth": Tree_base["max_depths"],
        "min_samples_split": Tree_base["min_samples_split"],
        "min_samples_leaf": Tree_base["min_samples_leaf"],
        "max_features" : Tree_base['max_features']
    },
    "AdaBoostRegressor" : {
        "n_estimators" : Tree_base["n_estimators"],
        "max_depth": Tree_base["max_depths"],
        "min_samples_split": Tree_base["min_samples_split"],
        "min_samples_leaf": Tree_base["min_samples_leaf"],
        "max_features" : Tree_base['max_features']
    },
    "ExtraTreesRegressor" : {
        "n_estimators" : Tree_base["n_estimators"],
        "max_depth": Tree_base["max_depths"],
        "min_samples_split": Tree_base["min_samples_split"],
        "min_samples_leaf": Tree_base["min_samples_leaf"],
        "max_features" : Tree_base['max_features']
    },
    "GradientBoostingRegressor" : {
        "n_estimators" : Tree_base["n_estimators"],
        "max_depth": Tree_base["max_depths"],
        "min_samples_split": Tree_base["min_samples_split"],
        "min_samples_leaf": Tree_base["min_samples_leaf"],
        "max_features" : Tree_base['max_features']
    }

}

CLASSIFICATION_HYPER = {
    "LogisticRegression" : {
        "penalty": ["l1", "l2"],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        "dual": [True, False]
    },
    "KNeighborsClassifier" : {
        "n_neighbors": range(1, 101),
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    },
    "RadiusNeighborsClassifier" : {
        "weights": ["uniform", "distance"],
        "algorithm" : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        "p": [1, 2]
    }, 
    "SVC" : {
        "kernel" : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        "epsilon": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    },

    "NuSVC" : {
        "nu" : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "kernel" : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    },
    "LinearSVC" : {
        "penalty": ["l1", "l2"],
        "loss": ["hinge", "squared_hinge"],
        "dual": [True, False],
        "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
    },
    "DecisionTreeClassifier" : {
        "criterion": ["gini", "entropy"],
        "max_depth": Tree_base['n_eatimators'],
        "min_samples_split": Tree_base['min_samples_split'],
        "min_samples_leaf": Tree_base['min_samples_leaf']
    },
    "ExtraTreeClassifier" : {
        "criterion": ["gini", "entropy"],
        "max_features": Tree_base["max_features"],
        "min_samples_split": Tree_base['min_samples_split'],
        "min_samples_leaf": Tree_base['min_samples_leaf']
        },
    "RandomForestClassifier":{
        "n_estimators": Tree_base["n_estimators"],
        "criterion": ["gini", "entropy"],
        "max_features": Tree_base["max_features"],
        "min_samples_split": Tree_base['min_samples_split'],
        "min_samples_leaf": Tree_base['min_samples_leaf']
    },
    "AdaBoostClassifier" : {
        "n_estimators": Tree_base["n_estimators"],
        "criterion": ["gini", "entropy"],
        "max_features": Tree_base["max_features"],
        "min_samples_split": Tree_base['min_samples_split'],
        "min_samples_leaf": Tree_base['min_samples_leaf']
    },
    "ExtraTreesClassifier" : {
        "n_estimators": Tree_base["n_estimators"],
        "criterion": ["gini", "entropy"],
        "max_features": Tree_base["max_features"],
        "min_samples_split": Tree_base['min_samples_split'],
        "min_samples_leaf": Tree_base['min_samples_leaf']
    },
    'GradientBoostingClassifier' : {
        "n_estimators": Tree_base["n_estimators"],
        "criterion": ["gini", "entropy"],
        "max_features": Tree_base["max_features"],
        "min_samples_split": Tree_base['min_samples_split'],
        "min_samples_leaf": Tree_base['min_samples_leaf'],
        "subsample" : np.arange(0.5,0.9,0.1)
    }
} 