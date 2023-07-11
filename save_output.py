import pandas as pd
import numpy as np
import pickle
# --------------------------- Save data and predictions --------------------------- #
class SaveOutput:
    """ 
    A calss object to save model outputs
    Parameters
    ----------
    train_id : pd.DataFrame
        Training set positional IDs
    eval_id : pd.DataFrame
        Evaluation set positional IDs
    test_id : pd.DataFrame
        Test set positional IDs
    out_features : str
        Name of the FHG coeficent
    custom_name: str
        A custom name for the running instance
    x_train : pd.DataFrame
        Predictor variables for training
    x_eval: pd.DataFrame 
        Predictor variables for evaluation
    test_x: pd.DataFrame    
        Predictor variables for testing
    y_train: np.array: 
        Target variables for training
    y_eval: np.array
        Target variables for evaluation
    test_y: np.array
        Target variables for testing 
    best_model: str
        Name of the best model
    loaded_model: any
        Model structure and weights 
    x_transform: bool
        Apply transformation to predictors
    y_transform: bool 
        Apply transformation to target 
    SI: bool
        Consider sientific system when plotting 
    """
    def __init__(self, train_id: pd.DataFrame, eval_id: pd.DataFrame,
                 test_id: pd.DataFrame, out_feature: str, custom_name: str,
                 x_train: pd.DataFrame, x_eval: pd.DataFrame, test_x: pd.DataFrame,
                 y_train: np.array, y_eval: np.array, test_y: np.array,
                 best_model: str, loaded_model: any, x_transform: bool, y_transform: bool,
                 val_method: str, SI: bool) -> None:
        
        self.train_id               = train_id
        self.eval_id                = eval_id
        self.test_id                = test_id
        self.out_feature            = out_feature
        self.custom_name            = custom_name
        self.x_train                = x_train
        self.x_eval                 = x_eval
        self.test_x                 = test_x
        self.y_train                = y_train
        self.y_eval                 = y_eval
        self.test_y                 = test_y
        self.best_model             = best_model
        self.loaded_model           = loaded_model
        self.y_trans                = y_transform
        self.x_trans                = x_transform
        self.SI                     = SI

        self.predictions_train      = 0
        self.predictions_valid      = 0
        self.predictions_test       = 0
        self.merged_data            = pd.DataFrame([])
        
    def processData(self) -> None:
        """ 
        A preprocessing step
        """
        # data transformation ----------------------------------------
        self.predictions_train = self.loaded_model.predict(self.x_train)
        self.predictions_valid = self.loaded_model.predict(self.x_eval)
        self.predictions_test = self.loaded_model.predict(self.test_x)

        # self.predictions_train_orig = self.predictions_train.copy()
        # self.predictions_valid_orig = self.predictions_valid.copy()
        # self.predictions_test_orig = self.predictions_test.copy()
        # self.y_train_orig = self.y_train.copy()
        # self.y_eval_orig = self.y_eval.copy()
        # self.test_y_orig = self.test_y.copy()

        if self.y_trans:
            t_y = pickle.load(open(self.custom_name+'/model/'+'train_y_'+self.out_features+'_tansformation.pkl', "rb"))
            self.predictions_train = t_y.inverse_transform(self.predictions_train.reshape(-1,1)).ravel()
            self.predictions_valid = t_y.inverse_transform(self.predictions_valid.reshape(-1,1)).ravel()
            self.predictions_test = t_y.inverse_transform(self.predictions_test.reshape(-1,1)).ravel()
            self.y_train = t_y.inverse_transform(self.y_train.reshape(-1,1)).ravel()
            self.y_eval = t_y.inverse_transform(self.y_eval.reshape(-1,1)).ravel()
            self.test_y = t_y.inverse_transform(self.test_y.reshape(-1,1)).ravel()

        if self.x_trans:
            t_x = pickle.load(open(self.custom_name+'/model/'+'train_x_'+self.out_features+'_tansformation.pkl', "rb"))
            col_names = self.x_train.columns
            self.x_train = t_x.inverse_transform(self.x_train)
            self.x_train = pd.DataFrame(data=self.x_train,
                                    columns=col_names).reset_index(drop=True)
            self.x_eval = t_x.inverse_transform(self.x_eval)
            self.x_eval = pd.DataFrame(data=self.x_eval,
                                    columns=col_names).reset_index(drop=True)
            self.test_x = t_x.inverse_transform(self.test_x)
            self.test_x = pd.DataFrame(data=self.test_x,
                                        columns=col_names).reset_index(drop=True)

        # ___________________________________________________
        # Build complete dataframe
        self.x_train['split'] = 'train'
        self.x_eval['split'] = 'eval'
        self.test_x['split'] = 'test'

        self.x_train['predicted'] = self.predictions_train
        self.x_eval['predicted'] = self.predictions_valid
        self.test_x['predicted'] = self.predictions_test

        self.x_train['target'] = self.y_train
        self.x_eval['target'] = self.y_eval
        self.test_x['target'] = self.test_y
        self.merged_data = pd.concat([self.x_train, self.x_eval, self.test_x], axis=0)

        if self.SI:
                self.merged_data['predicted'] = self.merged_data['predicted'] * 0.3048
                self.merged_data['target'] = self.merged_data['target'] * 0.3048
        
        # ___________________________________________________
        # Save dataframe
        self.merged_data.to_parquet(self.custom_name+'/metrics/'+str(self.custom_name)+'_'+self.best_model+'_'+self.out_feature+'.parquet')

        return
