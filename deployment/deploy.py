# Libraries 
# Catch warnings 
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from typing import Tuple
import data_loader as dataloader

# import comp_plot as pcml
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor, RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet, ARDRegression, OrthogonalMatchingPursuit
from sklearn.metrics import mean_absolute_error, max_error, explained_variance_score, mean_squared_error, mean_absolute_percentage_error
import lightgbm as ltb
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.neural_network import MLPRegressor


import tempfile
import pickle
import joblib
import json
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
from statsmodels.distributions.empirical_distribution import ECDF
import os
import sys 
from tqdm import tqdm
# library for parallel processing
from joblib import Parallel, delayed


# FHG
# ---------------------------initialize --------------------------- #
class DPModel:
    """ Main body of the machine learning model for estimating FHG
        Parameters

        Parameters
        ----------
        custom_name : str
            A custom name for the model to be extucuted 
    """
    def __init__(self) -> None:
        # os.chdir(b'/home/arash.rad/river_3d/conus-fhg/')
        pd.options.display.max_columns = 30
        self.rand_state                = 105

# --------------------------- Load trained models  --------------------------- #    
    def loadModel(self, out_feature: str, vote_flag: bool = False, meta_flag: bool = False,
                    best_flag: bool = True, file: str = 'bf', model_type: str = 'xgb',
                    ) -> Tuple[any, list]:
        """ Load the data and apply data filtering, transformation and 
        feature selection if nessassery

        Parameters
        ----------
        out_feature : str
            Name of the FHG coeficients
        x_transform : bool
            Whether to apply transformation to predictor variables or not 
            Opptions are:
            - True
            - False
        y_transform : bool
            Whether to apply transformation to target variable or not 
            Opptions are:
            - True
            - False
        R2_thresh : float
            The desired coeficent of determation to filter out bad measurments
            Opptions are:
            - any value between 0.0 - 100.0
        count_thresh: int
            The desired number of observations in each station to filter out bad measurments
        sample_type : str
            The type of predictor feature selection
            Opptions are:
            - "All": for considering all features
            - "Sub": for considering pre selected features
            - "test": a test case for unit testing
        pca: bool
            Whether to apply PCA or not 
            Opptions are:
            - True
            - False
        t_type: str
            type of transformation
            Opptions are:
            - log
            - power
            - quant
        
        Example
        --------
        >>> MlModel.loadData(out_feature = 'b', x_transform = False, 
                 y_transform = False, R2_thresh = 0.0,
                 sample_type = "Sub", pca = False, t_type = 'log')
        """
        
        # Load ML models
        if meta_flag:
            model = pickle.load(open('/model/'+file+'_'+out_feature+'_final_Meta_Model.pickle.dat', "rb"))
            best_model = pickle.load(open('/model/'+file+'_'+out_feature+'_'+model_type+'_final_Best_Model.pickle.dat', "rb"))
        elif vote_flag:
            model = pickle.load(open('/model/'+file+'_'+out_feature+'_final_Voting_Model.pickle.dat', "rb"))
            best_model = pickle.load(open('/model/'+file+'_'+out_feature+'_'+model_type+'_final_Best_Model.pickle.dat', "rb"))
        elif best_flag:
            model = best_model = pickle.load(open('/model/'+file+'_'+out_feature+'_'+model_type+'_final_Best_Model.pickle.dat', "rb"))
        
        # Extract feature names
        if model == 'xgb':
            feats = best_model.get_booster().feature_names
        elif model == 'lgb':
            feats = best_model.feature_name_
        else:
            feats = best_model.feature_name()
        return model, feats
    
    def process_target(self, dl_obj, target_name: str, vote_flag: bool=False, meta_flag: bool=False, 
                        best_flag: bool=True, file: str='bf', model_type: str='xgb'):
        if target_name.startswith("Y"):
            model_type = 'xgb'
        else:
            model_type = 'lgb'

        model, feats = loadModel(target_name, vote_flag=False, meta_flag=False, best_flag=True, file='bf', model_type=model_type)

        data_in = dl_obj.data[feats]
        y_pred_label = 'ML_' + target_name + '(m)'
        preds_all = model.predict(data_in)
        
        return y_pred_label, preds_all


# --------------------------- A driver class --------------------------- #           
class RunMlModel:
    @staticmethod
    def main(argv):
        """ The driver class to run ML model
        
        Parameters 
        ----------
        argv: list
            taken from bash script
        """
        nthreads     = int(argv[0])
        SI           = True

        # Load data
        dl_obj = dataloader.DataLoader(self.rand_state)
        dl_obj.readFiles()
        dl_obj.imputeData()
        dl_obj.buildPCA()

        # Load targets
        temp        = json.load(open('data/model_feature_names.json'))
        target_list = temp.get('out_features')
        del temp

        # Additional parameters
        vote_flag   = False
        meta_flag   = False
        best_flag   = True
        file        = 'bf'
        model_type  = 'xgb'
        log_y_t     = True
        
        deploy_obj = DPModel()

        # Parallelize the for loop
        results = Parallel(n_jobs=nthreads, backend="multiprocessing")(delayed(deploy_obj.process_target)(target_name, dl_obj, vote_flag, meta_flag, best_flag, file, model_type) for target_name in tqdm(target_list))
        # Unpack the results
        for y_pred_label, preds_all in results:
            if log_y_t:
                dl_obj.data[y_pred_label] = np.exp(preds_all)
            else:
                dl_obj.data[y_pred_label] = preds_all
            if SI:
                dl_obj.data[y_pred_label] = dl_obj.data[y_pred_label] * 0.3048
        target_list.append('comid')
        out_df = dl_obj.data[target_list]
        out_df.to_parquet('data/ml_exports.parquet')
        return

if __name__ == "__main__":
    RunMlModel.main([-1])
    # RunMlModel.main(sys.argv[1:])