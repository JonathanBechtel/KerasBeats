"""
NBeats model that is formalized in the following paper:  
    https://arxiv.org/abs/1905.10437
"""

import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model

### DIFFERENT BLOCK LAYERS:  GENERIC, SEASONAL, TREND
class GenericBlock(keras.layers.Layer):
    def __init__(self, 
                 lookback      = 7,
                 horizon       = 1, 
                 num_neurons   = 512,
                 block_layers  = 4):
        super(GenericBlock, self).__init__()
        """Generic Block for Nbeats model.  Inputs:  
            lookback: int -> Multiplier you use for horizon to determine
                             how big your training window is
            ----
            horizon:  int -> How far out into the future you would like
                             your predictions to be
            ----
            num_neurons: int -> How many layers to put into each Dense layer in
                                the generic block
            ----
            block_layers: int -> How many Dense layers to add to the block
        """
            
        # collection of layers in the block    
        self.layers_       = [keras.layers.Dense(num_neurons, activation = 'relu') 
                              for _ in range(block_layers)]
        self.lookback      = lookback
        self.horizon       = horizon
        
        # multiply lookback * forecast to get training window size
        self.backcast_size = horizon * lookback
        
        # numer of neurons to use for theta layer -- this layer
        # provides values to use for backcast + forecast in subsequent layers
        self.theta_size    = self.backcast_size + lookback
        
        # layer to connect to Dense layers at the end of the generic block
        self.theta         = keras.layers.Dense(self.theta_size, 
                                                use_bias = False, 
                                                activation = None)
        
    def call(self, inputs):
        # save the inputs
        x = inputs
        # connect each Dense layer to itself
        for layer in self.layers_:
            x = layer(x)
        # connect to Theta layer
        x = self.theta(x)
        # return backcast + forecast without any modifications
        return x[:, :self.backcast_size], x[:, -self.horizon:]
    
class TrendBlock(keras.layers.Layer):
    def __init__(self, 
                 lookback        = 7,
                 horizon         = 1,
                 num_neurons     = 512,
                 block_layers    = 4, 
                 polynomial_term = 2):
        super(TrendBlock, self).__init__()
        """Generic Block for Nbeats model.  Inputs:  
            lookback: int -> Multiplier you use for horizon to determine
                             how big your training window is
            ----
            horizon: int -> How far out into the future you would like
                             your predictions to be
            ----
            num_neurons: int -> How many layers to put into each Dense layer in
                                the generic block
            ----
            block_layers: int -> How many Dense layers to add to the block
            ----
            polynomial_term: int -> Degree of polynomial to use to understand
            trend term
            """
        self.polynomial_size = polynomial_term + 1
        self.layers_         = [keras.layers.Dense(num_neurons, 
                                                   activation = 'relu') 
                                for _ in range(block_layers)]
        self.lookback        = lookback
        self.horizon         = horizon
        self.theta_size      = 2 * (self.polynomial_size)
        self.backcast_size   = lookback * horizon
        self.theta           = keras.layers.Dense(self.theta_size, 
                                                  use_bias = False, 
                                                  activation = None)
        # taken from equation (2) in paper
        self.forecast_time   = K.concatenate([K.pow(K.arange(horizon, 
                                                             dtype = 'float') / horizon, i)[None, :]
                                 for i in range(self.polynomial_size)], axis = 0)
        self.backcast_time   = K.concatenate([K.pow(K.arange(self.backcast_size, 
                                                             dtype = 'float') / self.backcast_size, i)[None, :]
                                 for i in range(self.polynomial_size)], axis = 0)
    
    def call(self, inputs):
        x = inputs
        for layer in self.layers_:
            x = layer(x)
        x = self.theta(x)
        # create forecast / backcast from T / theta matrix
        backcast = K.dot(x[:, self.polynomial_size:], self.backcast_time)
        forecast = K.dot(x[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast
    
class SeasonalBlock(keras.layers.Layer):
    def __init__(self, 
                 lookback      = 7,
                 horizon       = 1,
                 num_neurons   = 512,
                 block_layers  = 4,
                 num_harmonics = 1):
        super(SeasonalBlock, self).__init__()
        """Seasonality Block for Nbeats model.  Inputs:  
            lookback: int -> Multiplier you use for horizon to determine
                             how big your training window is
            ----
            horizon: int -> How far out into the future you would like
                             your predictions to be
            ----
            num_neurons: int -> How many layers to put into each Dense layer in
                                the generic block
            ----
            block_layers: int -> How many Dense layers to add to the block
            ----
            num_harmonics: int -> The seasonal lag to use for your training window
        """
        self.layers_       = [keras.layers.Dense(num_neurons, 
                                                 activation = 'relu') 
                              for _ in range(block_layers)]
        self.lookback      = lookback
        self.horizon       = horizon
        self.num_harmonics = num_harmonics
        self.theta_size    = 4 * int(np.ceil(num_harmonics / 2 * horizon) - (num_harmonics - 1))
        self.backcast_size = lookback * horizon
        self.theta         = keras.layers.Dense(self.theta_size, 
                                                use_bias = False, 
                                                activation = None)
        self.frequency     = K.concatenate((K.zeros(1, dtype = 'float'), 
                             K.arange(num_harmonics, num_harmonics / 2 * horizon) / num_harmonics), 
                             axis = 0)

        self.backcast_grid = -2 * np.pi * (K.arange(self.backcast_size, dtype = 'float')[:, None] / self.backcast_size) * self.frequency

        self.forecast_grid = 2 * np.pi * (K.arange(horizon, dtype=np.float32)[:, None] / horizon) * self.frequency

        self.backcast_cos_template  = K.transpose(K.cos(self.backcast_grid))

        self.backcast_sin_template  = K.transpose(K.sin(self.backcast_grid))
        self.forecast_cos_template  = K.transpose(K.cos(self.forecast_grid))
        self.forecast_sin_template  = K.transpose(K.sin(self.forecast_grid))

    def call(self, inputs):
        x = inputs
        for layer in self.layers_:
            x = layer(x)
        x = self.theta(x)
        params_per_harmonic    = self.theta_size // 4
        backcast_harmonics_cos = K.dot(inputs[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = K.dot(x[:, 3 * params_per_harmonic:], 
                                       self.backcast_sin_template)
        backcast               = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = K.dot(x[:, :params_per_harmonic], 
                                       self.forecast_cos_template)
        forecast_harmonics_sin = K.dot(x[:, params_per_harmonic:2 * params_per_harmonic], 
                                       self.forecast_sin_template)
        forecast               = forecast_harmonics_sin + forecast_harmonics_cos
        return backcast, forecast
    
### CREATES NESTED LAYERS INTO A SINGLE NBEATS LAYER
class NBeats(keras.layers.Layer):
    def __init__(self,
                 model_type           = 'generic',
                 lookback             = 7,
                 horizon              = 1,
                 num_generic_neurons  = 512,
                 num_generic_stacks   = 30,
                 num_generic_layers   = 4,
                 num_trend_neurons    = 256,
                 num_trend_stacks     = 3,
                 num_trend_layers     = 4,
                 num_seasonal_neurons = 2048,
                 num_seasonal_stacks  = 3,
                 num_seasonal_layers  = 4,
                 num_harmonics        = 1,
                 polynomial_term      = 3,
                 **kwargs):
        super(NBeats, self).__init__()
        """Final N-Beats model that combines different blocks.  Inputs:
            model_type: str -> type of architecture to use.  Must be one of
                               ['generic', 'interpretable']
            ----
            lookback: int -> Multiplier you use for horizon to determine
                             how big your training window is
            ----
            horizon: int -> How far out into the future you would like
                             your predictions to be
            ----
            num_generic_neurons: int -> size of dense layers in generic block
            ----
            num_generic_stacks: int -> number of generic blocks to stack on top
                             of one another
            ----
            num_generic_layers: int -> number of dense layers to store inside a
                             generic block
            ----
            num_trend_neurons: int -> size of dense layers in trend block
            ----
            num_trend_stacks: int -> number of trend blocks to stack on top of
                             one another
            ----
            num_trend_layers: int -> number of Dense layers inside a trend block
            ----
            num_seasonal_neurons: int -> size of Dense layer in seasonal block
            ----
            num_seasonal_stacks: int -> number of seasonal blocks to stack on top
                             on top of one another
            ----
            num_seasonal_layers: int -> number of Dense layers inside a seasonal
                             block
            ----
            num_harmonics: int -> seasonal term to use for seasonal stack
            ----
            polynomial_term: int -> size of polynomial expansion for trend block
            """
        self.model_type           = model_type
        self.lookback             = lookback
        self.horizon              = horizon
        self.num_generic_neurons  = num_generic_neurons
        self.num_generic_stacks   = num_generic_stacks
        self.num_generic_layers   = num_generic_layers
        self.num_trend_neurons    = num_trend_neurons
        self.num_trend_stacks     = num_trend_stacks
        self.num_trend_layers     = num_trend_layers
        self.num_seasonal_neurons = num_seasonal_neurons
        self.num_seasonal_stacks  = num_seasonal_stacks
        self.num_seasonal_layers  = num_seasonal_layers
        self.num_harmonics        = num_harmonics
        self.polynomial_term      = polynomial_term
    
        # Model architecture is pretty simple: depending on model type, stack
        # appropriate number of blocks on top of one another
        # default values set from page 26, Table 18 from paper
        if model_type == 'generic':
            self.blocks_ = [GenericBlock(lookback       = lookback, 
                                         horizon        = horizon,
                                         num_neurons    = num_generic_neurons, 
                                         block_layers   = num_generic_layers)
                             for _ in range(num_generic_stacks)]
        if model_type == 'interpretable':
            self.blocks_ = [TrendBlock(lookback         = lookback,
                                       horizon          = horizon,
                                       num_neurons      = num_trend_neurons,
                                       block_layers     = num_trend_layers, 
                                       polynomial_term  = polynomial_term)
                            for _ in range(num_trend_stacks)] + [
                            SeasonalBlock(lookback      = lookback,
                                          horizon       = horizon,
                                          num_neurons   = num_seasonal_neurons,
                                          block_layers  = num_seasonal_layers,
                                          num_harmonics = num_harmonics)
                            for _ in range(num_seasonal_stacks)]
        
    def call(self, inputs):
        residuals = K.reverse(inputs, axes = 0)
        forecast  = inputs[:, -1:]
        for block in self.blocks_:
            backcast, block_forecast = block(residuals)
            residuals = keras.layers.Subtract()([residuals, backcast])
            forecast  = keras.layers.Add()([forecast, block_forecast])
        return forecast
    
### BUILDS AND COMPILES 
class NBeatsModel():
    
    def __init__(self, 
                 model_type:str           = 'generic',
                 lookback:int             = 7,
                 horizon:int              = 1,
                 num_generic_neurons:int  = 512,
                 num_generic_stacks:int   = 30,
                 num_generic_layers:int   = 4,
                 num_trend_neurons:int    = 256,
                 num_trend_stacks:int     = 3,
                 num_trend_layers:int     = 4,
                 num_seasonal_neurons:int = 2048,
                 num_seasonal_stacks:int  = 3,
                 num_seasonal_layers:int  = 4,
                 num_harmonics:int        = 1,
                 polynomial_term:int      = 3,
                 loss:str                 = 'mae',
                 learning_rate:float      = 0.001,
                 batch_size: int          = 1024):
        """Model used to create and initialize N-Beats model described in the following paper: 
           https://arxiv.org/abs/1905.10437
        
        Arguments (default listed in parentheses)
        -----------------------------------
        model: str -> what model architecture to use.  Must be one of ['generic', 'interpretable']
        ----
        lookback: int ->  what multiplier of the forecast size you want to use for your training window.
                              This number will be multiplied by the size of the horizon argument to get 
                              your training window size.  For example, if your forecast size is 3, and your lookback
                              is 4, your training window will be 4 * 3 = 12
        ----
        horizon: int -> How many steps into the future you want your model to predict.
        ----
        num_generic_neurons: int -> The number of neurons (columns) you want in each Dense layer for the generic block
        ----
        num_generic_stacks: int -> How many generic blocks to connect together
        ----
        num_generic_layers: int -> Within each generic block, how many dense layers do you want each one to have.  If
                                   you set this number to 4, and num_generic_neurons to 128, then you will have 4 Dense
                                   layers with 128 neurons in each one
        ----
        num_trend_neurons: int  -> Number of neurons to place within each Dense layer in each trend block
        ----
        num_trend_stacks: int -> number of trend blocks to stack on top of
                             one another
        ----
        num_trend_layers: int -> number of Dense layers inside a trend block
        ----
        num_seasonal_neurons: int -> size of Dense layer in seasonal block
        ----
        num_seasonal_stacks: int -> number of seasonal blocks to stack on top
                             on top of one another
        ----
        num_seasonal_layers: int -> number of Dense layers inside a seasonal
                             block
        ----
        num_harmonics: int -> seasonal term to use for seasonal stack
        ----
        polynomial_term: int -> size of polynomial expansion for trend block  
        ----
        loss: str -> what loss function to use inside keras.  accepts any
                     regression loss function built into keras.  You can find
                     more info here:  https://keras.io/api/losses/regression_losses/
        ----
        learning_rate: float -> learning rate to use when training the model
        ----
        batch_size: int -> batch size to use when training the model
        """
        self.model_type           = model_type
        self.lookback             = lookback
        self.horizon              = horizon
        self.num_generic_neurons  = num_generic_neurons
        self.num_generic_stacks   = num_generic_stacks
        self.num_generic_layers   = num_generic_layers
        self.num_trend_neurons    = num_trend_neurons
        self.num_trend_stacks     = num_trend_stacks
        self.num_trend_layers     = num_trend_layers
        self.num_seasonal_neurons = num_seasonal_neurons
        self.num_seasonal_stacks  = num_seasonal_stacks
        self.num_seasonal_layers  = num_seasonal_layers
        self.num_harmonics        = num_harmonics
        self.polynomial_term      = polynomial_term
        self.loss                 = loss
        self.learning_rate        = learning_rate
        self.batch_size           = batch_size
        
    def build_layer(self):
        """Initializes the Nested NBeats layer from initial parameters"""
        self.model_layer = NBeats(**self.__dict__)
        return self
        
    def build_model(self):
        """Creates keras model to use for fitting"""
        inputs     = keras.layers.Input(shape = (self.horizon * self.lookback, ), dtype = 'float')
        forecasts  = self.model_layer(inputs)
        self.model = Model(inputs, forecasts)
        return self
        
    def fit(self, X, y, **kwargs):
        """Build and fit model"""
        self.build_layer()
        self.build_model()
        self.model.compile(optimizer = keras.optimizers.Adam(self.learning_rate), 
                           loss      = [self.loss],
                           metrics   = ['mae', 'mape'])
        self.model.fit(X, y, batch_size = self.batch_size, **kwargs)
        return self
        
    def predict(self, X, **kwargs):
        """Passes predictions back to original keras layer"""
        return self.model.predict(X, **kwargs)
    
    
    def evaluate(self, y_true, y_pred, **kwargs):
        """Passes predicted and true labels back to the original keras model"""
        return self.model.evaluate(y_true, y_pred, **kwargs)
