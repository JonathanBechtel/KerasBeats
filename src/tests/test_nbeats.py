"""
Unit Tests For NBeats Models
"""
import pytest
from kerasbeats import utilities, NBeatsModel

class TestNBeatsConfiguration():
    
    def test_generic_prediction_output(self, numeric_data, generic_model):
        """Confirms that model works w/ univariate time series + prediction
        has the correct shape"""
        
        windows, labels = utilities.prep_time_series(numeric_data)
        
        generic_model.fit(windows, labels, epochs = 1)
        
        assert generic_model.predict(windows).shape == (windows.shape[0], 1)
        eval_ = generic_model.evaluate(windows, labels)
        assert type(eval_) == list
        assert len(eval_) == 3
        
        
    def test_interpretable_prediction_output(self, numeric_data,
                                             interpretable_model):
        """Confirms that model works w/ univariate timeries for interpretable
        model"""
        
        windows, labels = utilities.prep_time_series(numeric_data, 
                                                     lookback = 7,
                                                     horizon  = 2)
        interpretable_model.fit(windows, labels, epochs = 1)
        
        assert interpretable_model.predict(windows).shape == (windows.shape[0], 2)
        eval_ = interpretable_model.evaluate(windows, labels)
        assert type(eval_) == list
        assert len(eval_) == 3