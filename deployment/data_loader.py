# Libraries
import pandas as pd
import numpy as np
import pickle
import os
import json
import re
import fnmatch
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer


# Custom dataset
# --------------------------- Read data files --------------------------- #
class DataLoader:
    """ Main body of the data loader for preparing data for ML models

    Parameters
    ----------
    data_path : str
        The path to data that is used in ML model
    rand_state : int
        A random state number
    Example
    --------
    >>> DataLoader(rand_state = 105, data_path = 'data/input.parquet')
        
    """
    def __init__(self, rand_state: int, data_path: str = 'data/input.parquet') -> None:
        pd.options.display.max_columns  = 60
        self.data_path                  = data_path
        self.data                       = pd.DataFrame([])
        self.rand_state                 = rand_state
        np.random.seed(self.rand_state)
        # ___________________________________________________
        # Check directories
        if not os.path.isdir(os.path.join(os.getcwd(),"models/")):
            os.mkdir(os.path.join(os.getcwd(),"models/"))
        if not os.path.isdir(os.path.join(os.getcwd(),"data/")):
            os.mkdir(os.path.join(os.getcwd(),"data/"))
        if not os.path.isdir(os.path.join(os.getcwd(),'model_space/')):
            os.mkdir(os.path.join(os.getcwd(),'model_space/'))

    def readFiles(self) -> None:
        """ Read files from the directories
        """
        try:
            self.data = pd.read_parquet(self.data_path, engine='pyarrow')
        except:
            print('Wrong address or data format. Please use parquet file.')
        
        self.data = self.data[set(self.data.columns.to_list()) - set(['geometry','NHDFlowline','full_cats',
                                                                      'gridcode','number_unique_peaks','non_zero_years',
                                                                      'toCOMID','Hydroseq','RPUID','FromNode',
                                                                      'ToNode','VPUID','hy_cats','geometry_poly',
                                                                      'REACHCODE','sourcefc','comid','FEATUREID'])]
        # Find string columns (debug)
        # string_columns = []
        # # Iterate through each column and check if it contains string values
        # for col in self.data.columns:
        #     if self.data[col].dtype == 'O':  # 'O' represents object type (strings) in Pandas
        #         string_columns.append(col)  
        # print(string_columns)
        return

    # --------------------------- Add Binary Features --------------------------- #
    def addExtraFeatures(self, target_name: str) -> None:
        # Add VAA dummy
        self.data['vaa_dummy'] = self.data['roughness'].isnull().values
        self.data['vaa_dummy'] = self.data['vaa_dummy'] * 1

        # Add Scat dummy
        self.data['scat_dummy'] = self.data['BFICat'].isnull().values
        self.data['scat_dummy'] = self.data['scat_dummy'] * 1

        # Add discharge dummy
        if target_name.endswith("bf"):
            self.data['bf_ff'] = np.nan
            self.data['NWM'] = self.data['rp_2']
            self.data['discharge_dummy'] = 3
        else:
            self.data['in_ff'] = np.nan
            self.data['NWM'] = self.data['rp_1.5']
            self.data['discharge_dummy'] = 3
        return

    # --------------------------- Imputation --------------------------- #
    def imputeData(self) -> None:
        # Data imputation 
        impute = "median"
        if impute == "zero":
            self.data = self.data.fillna(-1) # a temporary brute force way to deal with NAN
        if impute == "median":
            self.data = self.data.replace(-1, np.nan)
            column_medians = self.data.median()
            self.data = self.data.fillna(column_medians)
        return

    # --------------------------- Dimention Reduction --------------------------- #     
    # PCA model
    def buildPCA(self, variable)  -> None:
        """ Builds a PCA and extracts new dimensions
        
        Parameters:
        ----------
        variable: str
            A string of target variable to be transformed

        Returns:
        ----------

        """
        matching_files = []
        folder_path = 'models'
        full_path = os.path.join(os.getcwd(), folder_path)
        # Iterate through the files in the folder
        for root, dirs, files in os.walk(full_path):
            for filename in files:
                # Check if both "PCA" and "Y_bf" are present in the file name
                search_pattern = f'*PCA*{variable}*'
                if all(fnmatch.fnmatch(filename, f'*{part}*') for part in search_pattern.split('*')):
                    matching_files.append(os.path.join(root, filename))

        # Extract the text between "PCA" and the 'vars' value using regular expressions
        pattern = f'{re.escape(variable)}(.*?)PCA'
        captured_texts = []
        for filename in matching_files:
            match = re.search(pattern, filename)
            if match:
                captured_texts.append(match.group(1))
            else:
                captured_texts.append("No match found")

        temp = json.load(open('model_space/dimension_space.json'))
        
        # Print the list of matching files
        for pca_item, text in zip(matching_files, captured_texts):
            pca =  pickle.load(open(pca_item, "rb"))
            temp_data = self.data[temp.get(text[1:-1])]
            new_data_pca = pca.transform(temp_data)
            for i in range(0, 5, 1):
                self.data[str(text[1:-1])+"_"+str(i)] = new_data_pca[:, i]

        return 
    
    # --------------------------- Data transformation --------------------------- #     
    # PCA model
    def transformXData(self, out_feature, data, t_type: str = 'power', x_transform: bool = False)  -> pd.DataFrame:
        """ Apply scaling and normalization to data
        
        Parameters:
        ----------
        variable: str
            A string of target variable to be transformed

        Returns:
        ----------

        """
        print('transforming and plotting ...')
        data = data.reset_index(drop=True)

        if x_transform:
            if t_type!='log':
                def applyScalerX(arr):
                    min_max_scaler = pickle.load(open('models/train_x_'+out_feature+'_scaler_tansformation.pkl', "rb"))
                    # min_max_scaler = pickle.load(open('/mnt/d/Lynker/R2_out/New/'+folder+'/conus-fhg/'+file+'/model/'+'train_'+arr.name+'_scaled.pkl', "rb"))
                    data_minmax = min_max_scaler.transform(arr.values.reshape(-1, 1))
                    return data_minmax.flatten()
                
                trans =  pickle.load(open('models/train_x_'+out_feature+'_tansformation.pkl', "rb"))
                scaler_data = data.apply(applyScalerX)
                trans_data = trans.transform(scaler_data)
                data = pd.DataFrame(trans_data, columns=data.columns)

            else:
                # Replace NA and inf
                data = np.log(np.abs(data)).fillna(0)
                data.replace([np.inf, -np.inf], -100, inplace=True)

        # Tests
        is_inf_data = data.isin([np.inf, -np.inf]).any().any()
        if is_inf_data:
            print('---- found inf in X {0} !!!!'.format(out_feature))
        has_missing_data  = data.isna().any().any()
        if has_missing_data:
            print('---- found nan in X {0} !!!!'.format(out_feature))

        return data
    
    def transformYData(self, out_feature, data, t_type: str = 'power', y_transform: bool = False)  -> np.array:
        """ Builds a PCA and extracts new dimensions
        
        Parameters:
        ----------
        variable: str
            A string of target variable to be transformed

        Returns:
        ----------

        """
        print('transforming and plotting ...')

        if y_transform:
            if t_type!='log':
                def applyScalerY(arr):
                    min_max_scaler = pickle.load(open('models/train_y_'+out_feature+'_scaler_tansformation.pkl', "rb"))
                    # min_max_scaler = pickle.load(open('/mnt/d/Lynker/R2_out/New/'+folder+'/conus-fhg/'+file+'/model/'+'train_'+arr.name+'_scaled.pkl', "rb"))
                    data_minmax = min_max_scaler.transform(arr.values.reshape(-1, 1))
                    return data_minmax.flatten()
                
                trans =  pickle.load(open('models/train_y_'+out_feature+'_tansformation.pkl', "rb"))
                #scaler_data = data.apply(applyScalerY)
                data = trans.transform(data)

            else:
                # Replace NA and inf
                data = np.log(np.abs(data)).fillna(0)
                data.replace([np.inf, -np.inf], -100, inplace=True)

        print('--------------- End of Y transformation ---------------')
        
        # Tests
        is_inf_y = np.isinf(data).any()
        if is_inf_y:
            print('---- found inf in Y {0} !!!!'.format(out_feature))
        
        has_missing_y = np.isnan(data).any()
        if has_missing_y:
            print('---- found nan in Y {0} !!!!'.format(out_feature))

        return data
        