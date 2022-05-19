import numpy as np
import pandas as pd
        
class cleaner_climate():  
    # Data cleaning script to calculate the number of heat days and other factors
    def __init__(self,  df_concat, df_reference, key):
        self.climate_city = df_concat
        self.reference_dist = df_reference
        self.key1 = key
        self.key2 = key.copy().remove('YEAR')
    
    def count_consecutive(self, x):
        # Count consecutive days
        x_str = ''.join(map(lambda x: str(int(x)), x.to_list())) # For each groupby, join the list into string
        x_str = x_str.replace('0', ' ').split(' ') # replace 0 with empty string and split
        x_str = list(filter(lambda x: len(x) >= 2, x_str)) # Filter to a list, count length > 2 (consecutive heatwave days)
        return len(''.join(x_str)) # Join all substrings together and count the total number of consecutive heatwave days within a month
    
    def monthly_stat(self):
        df_results = pd.DataFrame()
        df_groupby = self.climate_city.groupby(self.key1)
        df_results['TEMP_MEAN'] = df_groupby['T2M_MAX'].mean() # Mean of daily temperature
        df_results['TEMP_RNG'] = df_groupby['T2M_RANGE'].max() # Maximum daily temperature range
        return df_results
    
    def featurize(self):
        df_monthly_stat = self.monthly_stat().reset_index().merge(self.reference_dist, how = 'left', on = self.key2)
        df_results = self.climate_city.merge(df_monthly_stat, how = 'left', on = self.key1)
        # ABOVE_LIMIT: If daily max temperature > 90th percentile from the reference, 1. Else 0
        df_results['ABOVE_LIMIT'] = (df_results['T2M_MAX'] > df_results['TEMPMAX_90th']).astype(np.float64)
        df_monthly_stat['HEAT_DAYS'] = df_results.groupby(self.key1)['ABOVE_LIMIT'].agg(self.count_consecutive).values
        return df_monthly_stat.drop(['TEMPMAX_90th'], axis = 1)
