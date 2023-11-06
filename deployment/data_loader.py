# Libraries
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer
from sklearn.decomposition import PCA
import scipy
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os
import json

# FHG dataset
# --------------------------- Read data files --------------------------- #
class DataLoader:
    """ Main body of the data loader for preparing data for ML models

    Parameters
    ----------
    data_path : str
        The path to data that is used in ML model
    target_data_path : str
        The path to target widt/depth data that is used in ML model
    rand_state : int
        A random state number
    out_feature : str
        The name of the FHG coeficent to be used
    custom_name : str
        A custom name defiend by user to name modeling task
    x_transform : str
        Whether to apply transformation to predictor variables or not 
        Opptions are:
        - True
        - False
    x_transform : str
        Whether to apply transformation to predictor variables or not 
        Opptions are:
        - True
        - False
        - defaults to False
    y_transform : bool
        Whether to apply transformation to target variable or not 
        Opptions are:
        - True
        - False
        - defaults to False
    R2_thresh : float
        The desired coeficent of determation to filter out bad measurments
        Opptions are:
        - any value between 0.0 - 100.0
        - defaults to 0.0
    count_thresh: int
            The desired number of observations in each station to filter out bad measurments
    Example
    --------
    >>> DataLoader(data_path = 'data/test.parquet', out_feature = 'b', rand_state = 115,
        custom_name = 'test', x_transform = False, y_transform = False, R2_thresh = 0.0, count_thresh = 3)
        
    """
    def __init__(self, rand_state: int, data_path: str = 'data/input.parquet') -> None:
        pd.options.display.max_columns  = 60
        self.data_path                  = data_path
        self.data                       = pd.DataFrame([])
        self.rand_state                 = rand_state
        np.random.seed(self.rand_state)
        
        # ___________________________________________________
        # Check directories
        if not os.path.isdir(os.path.join(os.getcwd(),self.custom_name,"model/")):
            os.mkdir(os.path.join(os.getcwd(),self.custom_name,"model/"))
        if not os.path.isdir(os.path.join(os.getcwd(),self.custom_name,"data/")):
            os.mkdir(os.path.join(os.getcwd(),self.custom_name,"data/"))
        if not os.path.isdir(os.path.join(os.getcwd(),self.custom_name,'model_space/')):
            os.mkdir(os.path.join(os.getcwd(),self.custom_name,'model_space/'))

    def readFiles(self) -> None:
        """ Read files from the directories
        """
        try:
            self.data = pd.read_parquet(self.data_path, engine='pyarrow')
        except:
            print('Wrong address or data format. Please use parquet file.')
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
    def buildPCA(self)  -> None:
        """ Builds a PCA and extracts new dimensions
        
        Parameters:
        ----------
        feat_list: list
            A list containing all feature names to be reduced 
        n_components: int, default=None    
            Number of components to keep. if n_components is not set all components are kept
        name: str    
            Reduced feature names

        Returns:
        ----------

        """
        matching_files = []
        folder_path = '/models/'
        # Iterate through the files in the folder
        for root, dirs, files in os.walk(folder_path):
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

        temp = json.load(open('/model_space/dimension_space.json'))
        
        # Print the list of matching files
        for pca_item, text in zip(matching_files, captured_texts):
            pca =  pickle.load(open(pca_item, "rb"))
            temp_data = self.data[temp.get(text[1:-1])]
            new_data_pca = pca.transform(temp_data)
            for i in range(0, 5, 1):
                self.data[str(text[1:-1])+"_"+str(i)] = new_data_pca[:, i]

        print("\n ------------- End of dimension reduction ----------- \n")
        return 
        