"""
Tests splitting functions to make sure they prepare data correctly
"""
import pytest
import numpy as np
from kerasbeats import utilities

class TestUnivariateTimeSeriesSplit():
    
    def test_single_horizon(self, numeric_data):
        """test splitter with different inputs"""
        windows, labels = utilities.prep_time_series(numeric_data, 
                                                     lookback = 5,
                                                     horizon  = 1)
        
        assert (windows == [[ 0,  1,  2,  3,  4],
                           [ 1,  2,  3,  4,  5],
                           [ 2,  3,  4,  5,  6],
                           [ 3,  4,  5,  6,  7],
                           [ 4,  5,  6,  7,  8],
                           [ 5,  6,  7,  8,  9],
                           [ 6,  7,  8,  9, 10],
                           [ 7,  8,  9, 10, 11],
                           [ 8,  9, 10, 11, 12],
                           [ 9, 10, 11, 12, 13],
                           [10, 11, 12, 13, 14],
                           [11, 12, 13, 14, 15],
                           [12, 13, 14, 15, 16],
                           [13, 14, 15, 16, 17],
                           [14, 15, 16, 17, 18]]).all()
        
        assert (labels == [[ 5],
                          [ 6],
                          [ 7],
                          [ 8],
                          [ 9],
                          [10],
                          [11],
                          [12],
                          [13],
                          [14],
                          [15],
                          [16],
                          [17],
                          [18],
                          [19]]).all()
        
    def test_single_horizon_w_list_input(self, numeric_data):
        """Tests to make sure splitter works with raw list as input"""
        windows, labels = utilities.prep_time_series(numeric_data.tolist(), 
                                                     lookback = 5,
                                                     horizon  = 1)
        
        assert (windows == [[ 0,  1,  2,  3,  4],
                           [ 1,  2,  3,  4,  5],
                           [ 2,  3,  4,  5,  6],
                           [ 3,  4,  5,  6,  7],
                           [ 4,  5,  6,  7,  8],
                           [ 5,  6,  7,  8,  9],
                           [ 6,  7,  8,  9, 10],
                           [ 7,  8,  9, 10, 11],
                           [ 8,  9, 10, 11, 12],
                           [ 9, 10, 11, 12, 13],
                           [10, 11, 12, 13, 14],
                           [11, 12, 13, 14, 15],
                           [12, 13, 14, 15, 16],
                           [13, 14, 15, 16, 17],
                           [14, 15, 16, 17, 18]]).all()
        
        assert (labels == [[ 5],
                          [ 6],
                          [ 7],
                          [ 8],
                          [ 9],
                          [10],
                          [11],
                          [12],
                          [13],
                          [14],
                          [15],
                          [16],
                          [17],
                          [18],
                          [19]]).all()
        
    def test_multidimensional_data_throws_error(self, nested_time_series):
        """Makes sure that code throws up error message w/ invalid input"""
        try:
            windows, labels = utilities.prep_time_series(nested_time_series, 
                                                     lookback = 5,
                                                     horizon  = 1)
        except Exception as e:
            assert str(e) == """Input should be a univariate time 
                                       series with only a single column"""
        
    def test_multiple_horizons(self, numeric_data):
        """Test same splitter function, but with a multi step horizon"""
        windows, labels = utilities.prep_time_series(numeric_data, 
                                                     lookback = 5,
                                                     horizon  = 2)
        
        assert (windows == [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
                            [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
                            [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
                            [ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                            [ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13],
                            [ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
                            [ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
                            [ 7,  8,  9, 10, 11, 12, 13, 14, 15, 16],
                            [ 8,  9, 10, 11, 12, 13, 14, 15, 16, 17]]).all()
        
        assert (labels == [[10, 11],
                           [11, 12],
                           [12, 13],
                           [13, 14],
                           [14, 15],
                           [15, 16],
                           [16, 17],
                           [17, 18],
                           [18, 19]]).all()
        

        
class TestMultivariateTimeSeriesSplit():
    """Used to test time series splitter that combines multiple time series"""
    
    def test_single_horizon(self, nested_time_series):
        windows, labels = utilities.prep_multiple_time_series(nested_time_series,
                                    label_col = 'label',
                                    data_col = 'value',
                                    lookback = 5, 
                                    horizon = 1)
        
        a = nested_time_series.loc[nested_time_series.label == 'a', 'value']
        b = nested_time_series.loc[nested_time_series.label == 'b', 'value']
        
        win_a, lab_a = utilities.prep_time_series(a, 
                                                  lookback = 5,
                                                  horizon  = 1)
        
        win_b, lab_b = utilities.prep_time_series(b, 
                                                  lookback = 5,
                                                  horizon  = 1)
        
        assert (np.vstack([win_a, win_b]) == windows).all()
        
        assert (np.vstack([lab_a, lab_b]) == labels).all()
    