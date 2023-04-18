# Liberaries
import datetime
import pyarrow.parquet as pq
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np

class FilterADCP:
    """     
    Filtering class to read HYDRoSWOT data and perform a filtering on it. The filtering  
    of data (``pandas dataframe``) involves droping rows containg nan i measurments of
    depth and width as well as and limitting observations to those of after 2010
    to be consistent with Sentianl data.    
    """
    def __init__(self, year):
        self.year = str(year)
        """ 
        Parameters
        ----------
        year : string
          
        return_lr : empty
            It saves the name and corrdiante of stations.                                                                
        """

    def filterData(self):
        # read ADCP and ignor timestamp
        adcp = pq.read_table('/mnt/d/Lynker/FEMA_HECRAS/bankfull_W_D/data/adcp.parquet').to_pandas(safe=False)
        print('Number of unique datums: {0}'.format(adcp['coord_datum_cd'].unique()))
        print('Visual check for timestamps: {0}\n'.format( adcp['site_visit_start_dt'].min()))

        # Remove NAN
        adcp_nona = adcp.dropna(subset=['site_no','dec_lat_va','dec_long_va','stream_wdth_va','max_depth_va'])
        adcp_nona = adcp_nona[['site_no','site_visit_start_dt',
                            'q_meas_dt','station_nm','dec_lat_va','dec_long_va',
                            'coord_datum_cd','meas_q_va','stream_wdth_va','max_depth_va']]
        print('ADCP shape after NAN removal: {0}\n'.format(adcp_nona.shape))
        
        # Filter date
        adcp_nona['date'] =  pd.to_datetime(adcp_nona['site_visit_start_dt'], format='%Y-%m-%d')
        adcp_nona['date'] = adcp_nona['date'].dt.date
        adcp_nona['date'] =  pd.to_datetime(adcp_nona['date'], format='%Y-%m-%d')
        adcp_date = adcp_nona.loc[(adcp_nona['date']>=self.year+'-01-01')]
                                #   & (adcp_nona['date']<'2011-01-01')]  
        adcp_date = adcp_date[['site_no','date',
                            'station_nm','dec_lat_va','dec_long_va',
                            'meas_q_va','stream_wdth_va','max_depth_va']]
        print('ADCP shape after filttering time: {0}'.format(adcp_date.shape))

        # Groupby 
        agg_adcp = adcp_date.groupby('site_no').max().reset_index().rename(columns={'site_no': 'siteID', 'dec_lat_va': 'lat', 'dec_long_va':'long'})
        agg_adcp = agg_adcp[['siteID','lat','long','station_nm']]
        print(agg_adcp.head())

        # Just CONUS
        adcp_gdf = gpd.GeoDataFrame(
            agg_adcp, geometry=gpd.points_from_xy(agg_adcp.long, agg_adcp.lat))
        lat_point_list = [22.02, 49.84, 47.75, 23.8, 22.02]
        lon_point_list = [-127.56, -128.79, -61.47, -69.64, -127.56]
        boundary_geom = Polygon(zip(lon_point_list, lat_point_list))
        adcp_gdf = adcp_gdf[adcp_gdf.geometry.within(boundary_geom)]
        print('Number of unique stations: {0}\n'.format(len(adcp_gdf.site_no.unique())))
        adcp_gdf = adcp_gdf[['siteID','lat','long','station_nm']]

        # Write 
        adcp_gdf.to_parquet('data/station_ids.parquet')
        print('wrote to ``station_ids.parquet``')

class RunFiltering:
    @staticmethod
    def main():
        year = str(input("Enter your filtering year date: "))
        filter_obj = FilterADCP(year)
        filter_obj.filterData()

if __name__ == "__main__":
    RunFiltering.main()


