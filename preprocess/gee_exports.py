# Liberaries
import pandas as pd
import numpy as np
import glob
import os

class FilterGee:
    """     
    This is to read and filter GEE exported files and prepare them for 
    the ML model 
    """
    def __init__(self):
        """ 
        ----------                                                               
        """

    def readGEE(self):
        # read the GEE data
        cwd = os.getcwd()
        all_files = glob.glob(os.path.join(cwd , 'saved/GEEdata/ff_gee_exports2/*.csv'))

        list = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0, dtype={
                            'siteID': 'str',
                        })
            list.append(df)

        frame = pd.concat(list, axis=0, ignore_index=True)
        # Remove duplicates
        frame = frame.drop_duplicates(subset=['siteID']).reset_index(drop=True)

        # Drop useless features and add lat and long 
        frame['long'] = frame['.geo'].map(lambda a: eval(a).get('coordinates')[0])
        frame['lat'] = frame['.geo'].map(lambda a: eval(a).get('coordinates')[1])
        frame = frame.drop(['.geo', 'system:index'], axis=1)
        # Write to file
        frame.to_parquet('data/gee_init.parquet')


class RunFiltering:
    @staticmethod
    def main():
        filter_obj = FilterGee()
        filter_obj.readGEE()

if __name__ == "__main__":
    RunFiltering.main()


