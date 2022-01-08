"""
Helper functions for using the NBeatsmodel
"""

from pandas import DataFrame, Series
import numpy as np

class InvalidArgumentError(Exception):
    """Used to validate user input"""
    pass

def prep_time_series(data, 
                     lookback:int = 7, 
                     horizon:int = 1) -> (np.ndarray, np.ndarray):
    """
    Creates windows and their corresponding labels for each unique time series
    in a dataset
    
    E.g. if horizon = 2 and lookback = 3
    Input: [1, 2, 3, 4, 5, 6, 7, 8] -> Output: ([1, 2, 3, 4, 5, 6], [7, 8])
    
    Training window goes back by 3 * 2 values
    
    inputs:
        
        :param data:  univariate time series you want to create windows for.  Can be pandas dataframe, numpy array or list
        :param lookback:  multiple of forecast horizon that you want to use for training window
        :param horizon: how far out into the future you want to predict
        
    :returns: tuple with data types: (np.ndarray, np.ndarray) containing training windows and labels
    """
    
    ### convert data into numpy array, if necessary
    if type(data) == list:
        data = np.array(data)
        
    if type(data) in [DataFrame, Series]:
        data = data.values
        
    if data.ndim > 1:
        if data.shape[1] > 1:
            raise InvalidArgumentError("""Input should be a univariate time 
                                       series with only a single column""")
    
    # size of training window
    backcast_size = lookback * horizon
    
    # total length of data for training window + horizon
    window_step = np.expand_dims(np.arange(backcast_size + horizon), 
                                 axis=0)
        
    # creates index values for data
    window_indexes = window_step + np.expand_dims(
        np.arange(len(data) - (backcast_size + horizon - 1)), axis=0).T
    
    windowed_array = data[window_indexes]
    
    return windowed_array[:, :-horizon], windowed_array[:, -horizon:]

def prep_multiple_time_series(data, 
                             label_col: str,
                             data_col: str,
                             lookback: int = 7,
                             horizon: int = 1) -> (np.ndarray, np.ndarray):
    """
    Creates training windows for time series that are stacked on top of each 
    other
    
    example:
        
    inputs =  [['ar', 1]
               ['ar', 2],
               ['ar', 3],
               ['br', 5],
               ['br', 6],
               ['br', 7]]
        
    outputs = [[1, 2],   [[3],
              [5, 6]],   [7]]
        
    It treats the values associated with 'ar' and 'br' as separate time series
        
    inputs:
        
        :param data:  pandas DataFrame that has at least two columns, one that are labels for each unique time series in your dataset, and another that are the timeseries values
        :param label_col: the name of the column that labels each time series
        :param data_col:  the column that contains the time series values
        :param lookback:  what multiple of your horizon you want your training data to be eg -- a horizon of 2 and lookback of 5 creates a training window of 10
        :param horizon:   how far into the future you want to predict
        
    :returns: tuple with data types: (np.ndarray, np.ndarray) containing training windows and labels for the concatenated time series
    """
    # will be used to contain each unique time series inside the dataset
    ts_windows = []
    ts_vals    = []
    
    # labels for each time series within dataset
    unique_ts = data[label_col].unique()
    
    # create windows + labels for each timeseries in the dataset
    for label in unique_ts:
        query = data[label_col] == label
        tmp = data.loc[query, data_col].values
        windows, labels = prep_time_series(tmp, lookback, horizon)
        ts_windows.append(windows)
        ts_vals.append(labels)

    return np.vstack(ts_windows), np.vstack(ts_vals)    
