"""
Creates test fixtures to test neural network and utility functions
"""
import pytest
import numpy as np
import pandas as pd
from kerasbeats import NBeatsModel

# single univariate time series values
@pytest.fixture(scope='function')
def numeric_data():
    return np.arange(20)

# nested time series
@pytest.fixture(scope='function')
def nested_time_series():
    df = pd.DataFrame()
    df['label'] = ['a'] * 10 + ['b'] * 10
    df['value'] = [i for i in range(10)] * 2
    return df

# generic model
@pytest.fixture(scope='session')
def generic_model():
    model = NBeatsModel(model_type = 'generic',
                        batch_size = 5)
    return model

# interpretable model
@pytest.fixture(scope='session')
def interpretable_model():
    model = NBeatsModel(model_type = 'interpretable',
                        batch_size = 5,
                        horizon = 2)
    return model