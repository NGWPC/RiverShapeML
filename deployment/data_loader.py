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
        dump_list = ['R2', 'siteID']
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

        return data
    
    def transformYData(self, out_feature, data, t_type: str = 'power', x_transform: bool = False, y_transform: bool = False)  -> pd.DataFrame:
        """ Builds a PCA and extracts new dimensions
        
        Parameters:
        ----------
        variable: str
            A string of target variable to be transformed

        Returns:
        ----------

        """
        print('transforming and plotting ...')
        dump_list = ['R2', 'siteID']
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

        return data
        if y_transform:
            if t_type!='log':
                def applyScalerY(arr):
                    min_max_scaler = pickle.load(open('models/train_y_'+out_feature+'_scaler_tansformation.pkl', "rb"))
                    # min_max_scaler = pickle.load(open('/mnt/d/Lynker/R2_out/New/'+folder+'/conus-fhg/'+file+'/model/'+'train_'+arr.name+'_scaled.pkl', "rb"))
                    data_minmax = min_max_scaler.transform(arr.values.reshape(-1, 1))
                    return data_minmax.flatten()
                
                trans =  pickle.load(open('models/train_y_'+out_feature+'_tansformation.pkl', "rb"))
                trans_data = trans.transform(scaler_data)

                # scaler_y = StandardScaler()
                train_y = self.train[[self.out_feature]].reset_index(drop=True)
                train_y_cp = train_y.copy()
                train_y_t = t_y.fit_transform(train_y)
                pickle.dump(t_y, open(self.custom_name+'/model/'+'train_y_'+self.out_feature+'_tansformation.pkl', "wb"))
                # train_y_pt = scaler_x.fit_transform(train_y_pt)
                train_y = train_y_t.ravel()
                if plot_dist:
                    self.plotDist(train_y_cp, pd.DataFrame({self.out_feature: train_y}), 'train')

                test_y = self.test[[self.out_feature]].reset_index(drop=True)
                test_y_cp = test_y.copy()
                test_y_t = t_y.transform(test_y)
                # test_y_pt = scaler_y.transform(test_y_pt)
                test_y = test_y_t.ravel()
                if plot_dist:
                    self.plotDist(test_y_cp, pd.DataFrame({self.out_feature: test_y}), 'test')
            else:
                train_y = self.train[[self.out_feature]].reset_index(drop=True)
                train_y_cp = train_y.copy()
                # Replace NA and inf
                train_y = np.log(np.abs(train_y)).fillna(0)
                train_y.replace([np.inf, -np.inf], -100, inplace=True)
                if plot_dist:
                    self.plotDist(train_y_cp, train_y, 'train')
                train_y = train_y.values.ravel()

                test_y = self.test[[self.out_feature]].reset_index(drop=True)
                test_y_cp = test_y.copy()
                # Replace NA and inf
                test_y = np.log(np.abs(test_y)).fillna(0)
                test_y.replace([np.inf, -np.inf], -100, inplace=True)
                if plot_dist:
                    self.plotDist(train_y_cp, test_y, 'test')
                test_y = test_y.values.ravel()
        else:
            train_y = self.train[[self.out_feature]].reset_index(drop=True)
            train_y = train_y.values.ravel()
            test_y = self.test[[self.out_feature]].reset_index(drop=True)
            test_y = test_y.to_numpy().reshape((-1,))

        print('--------------- End of transformation ---------------')
        
        # Test data
        is_inf_train_x = train_x.isin([np.inf, -np.inf]).any().any()
        if is_inf_train_x:
            print('---- found inf in train x !!!!' )
        is_inf_test_x = test_x.isin([np.inf, -np.inf]).any().any()
        if is_inf_test_x:
            print('---- found inf in test x !!!!' )
        is_inf_train_y = np.isinf(train_y).any()
        if is_inf_train_y:
            print('---- found inf in train y !!!!' )
        is_inf_test_y = np.isinf(test_y).any()
        if is_inf_test_y:
            print('---- found inf in test y !!!!' )
        has_missing_train_x = train_x.isna().any().any()
        if has_missing_train_x:
            print('---- found nan in train x !!!!' )
        has_missing_test_x= test_x.isna().any().any()
        if has_missing_test_x:
            print('---- found nan in test x !!!!' )
        has_missing_train_y = np.isnan(train_y).any()
        if has_missing_train_y:
            print('---- found nan in train y !!!!' )
        has_missing_test_y = np.isnan(test_y).any()
        if has_missing_test_y:
            print('---- found nan in test y !!!!' )


        return 
        