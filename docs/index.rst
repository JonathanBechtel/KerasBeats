KerasBeats
------------
An easy, accessible way to use the NBeats model architecture in Keras.

.. image:: https://raw.githubusercontent.com/JonathanBechtel/KerasBeats/main/common/images/nbeats.PNG

Introduction
------------

The motivation for this project was to take the NBeats model architecture defined in the original paper here:  https://arxiv.org/abs/1905.10437 and reproduce it in a widely accessible form in keras.  In the past few years this model has become very popular as a timer series forecasting tool, but its implementation in keras seemed elusive, without an easy-to-use, well documented option online that'd be simple for newcomers to try.   This package allows anyone with basic knowledge of keras to be able to quickly use NBeats in its generic and interpretable form with just a few function calls. 

To that end, KerasBeats was built with the following ideas in mind:
 - It should reflect the original model architecture as closely as possible.
 - It should have a simple, high level architecture that allows people to get started as quickly as possible using the familar :code:`fit/predict` methods that everyone is already familiar with
 - It should allow you to quickly and easily use it as a keras model to take advantage of the libraries existing functionality and enable its use in existing workflows
 
Installation
------------
kerasbeats can be installed with the following line: 

::

    pip install keras-beats

Basic Usage
-----------

The N-Beats model architecture assumes that you take a univariate time series and create training data that contains previous values for an observation at a particular point in time.  For example, let's assume you have the following univariate time series:

::

    # sample time series values
    time_vals = [1, 2, 3, 4, 5]

If you were predicting one period ahead and wanted to use the previous two values in the time series as input, you want your data to be formatted like this:

::

    # data formatting for N-beats
    # each row represents the previous two values for the currently observed one
    X = [[1, 2],
         [2, 3],
         [3, 4]]
     
    y = [[3], 
         [4], 
         [5]]

The idea here is that :code:`[1, 2]` were the two values that preceded :code:`3`, :code:`[2, 3]` were the two that preceeded :code:`4`, and so on.  

Once your input data is formatted like this then you can use :code:`kerasbeats` in the following way:

::

    from kerasbeats import NBeatsModel
    mod = NBeatsModel()
    mod.fit(X, y)

When you are finished fitting your model you can use the :code:`predict` and :code:`evaluate` methods, which are just wrappers on the original keras methods, and would work in the same way.

Data Prep
---------

Most time series data typically comes in column format, so a little data prep is usually needed before you can feed it into :code:`kerasbeats`. You can easily do this yourself, but there are some built in functions in the :code:`kerasbeats` package to make this a little easier.  

Univariate Time Series
----------------------

If you have a single time series, you can use the :code:`prep_time_series` function to get your data in the appropriate format.  It works like this:

::

    from kerasbeats import prep_time_series
    # sample data:  a mock time series with ten values
    time_vals = np.arange(10)
    windows, labels = prep_time_series(lookback = 5, horizon = 1)

Once you are done with this the value of :code:`windows` will be the following numpy array:

::

    # training window of 5 values
    array([[0, 1, 2, 3, 4],
           [1, 2, 3, 4, 5],
           [2, 3, 4, 5, 6],
           [3, 4, 5, 6, 7],
           [4, 5, 6, 7, 8]])
       
The value of `labels` will be the following numpy array:

::

    # the value that followed the preceeding 5
    array([[5],
           [6],
           [7],
           [8],
           [9]])
 
This method accepts numpy arrays, lists, and pandas Series and DataFrames as input, but they must be one column if they are not then you'll receive an error message.
 
The function contains two separate arguments:
 
 - **horizon:** how far out into the future you want to predict.  A horizon value of 1 means you are predicting one step ahead. A value of two means you are predicting two steps ahead, and so on
 - **lookback:** what multiple of the `horizon` you want to use for training data.  So if `horizon` is 1 and `lookback` is 5, your training window will be the previous 5 values.  If `horizon` is 2 and `lookback` is 5, then your training window will be the previous 10 values.
 
Multiple Time Series
--------------------
 
You could conceivably use `kerasbeats` to learn a combination of time series jointly, assuming they shared common patterns between them.  
 
For example, here's a simple dataset that contains two different time series in a dataframe:
 
::
 
     import pandas as pd

     df = pd.DataFrame()
     df['label'] = ['a'] * 10 + ['b'] * 10
     df['value'] = [i for i in range(10)] * 2
 
:code:`df` would look like this in a jupyter notebook:

.. image:: https://raw.githubusercontent.com/JonathanBechtel/KerasBeats/main/common/images/sample_df.PNG
 
This contains two separate time series, one for value :code:`a`, and another for value :code:`b`.  If you want to prep your data so each time series for each label is turned into its corresponding training windows and labels you can use the :code:`prep_multiple_time_series` function:
 
::
 
     from kerasbeats import prep_multiple_time_series
     windows, labels = prep_multiple_time_series(df, label_col = 'label', data_col = 'value', lookback = 5, horizon = 2)

This function will perform the :code:`prep_time_series` function for each unique value specified in the :code:`label_col` column and then concatenate them together in the end, and you can then pass :code:`windows` and :code:`labels` into the :code:`NBeatsModel`.
     
KerasBeats layer
----------------

The :code:`NBeatsModel` is an abstraction over a functional keras model.  You may just want to use the underlying keras primitives in your own work without the very top of the model itself.  

The basic building block of :code:`kerasbeats` is a custom keras layer that contains all of the N-Beats blocks stacked together.  If you want access to this layer directly you can call the :code:`build_layer` method:

::

    from kerasbeats import NBeatsModel
    model = NBeatsModel()
    model.build_layer()

This exposes the :code:`layer` attribute, which is a keras layer that can be re-used in larger, multi-faceted models if you would like.

Using KerasBeats as a keras model
---------------------------------

Likewise, you may want to access some underlying keras functionality that's not directly available in :code:`NBeatsModel`.  In particular, when you call :code:`fit` using the :code:`NBeatsModel` wrapper, the :code:`compile` step is done for you automatically.  

However, if you wanted to define your own separate loss functions, or define callbacks, you can access the fully built keras model in the following way:

::

    nbeats = NBeatsModel()
    nbeats.build_layer()
    nbeats.build_model()

After these two lines, you can access the :code:`model` attribute, which will give you access to the full keras model.

So if you wanted to specify a different loss function or optimizer, you could do so easily:

::

    nbeats.model.compile(loss = 'mse',
                         optimizer = tf.keras.optimizers.RMSProp(0.001))
    nbeats.model.fit(windows, labels)

Please note that if you want to use the underlying keras model directly, you should use :code:`nbeats.model.fit()` and not :code:`nbeats.fit`, since it will try and compile the model for you automatically after you call it.

API Reference
-------------

Important Caveats
#################

The following sections describe the different functions and classes available in :code:`kerasbeats`.  
A few important notes:
 - All of the default values are designed to mimic what was originally in the paper, but are not necessarily best for your project
 - batch size and learning rate are parameters you can usually specify inside keras itself, but they are defined at initialization because these values were explicitly mentioned in the paper.  If you want to set them yourself, you should call :code:`NBeatsModel().build_layer().build_model()` in order to set these values in keras directly.
 - If you are going to use the interpretable model, you should probably set :code:`horizon` to at least 2.  Due to the way some other parameters are specified, the matrix math typically only works if you are predicting more than one day out.

NBeatsModel
###########

.. module:: nbeats
.. autoclass:: NBeatsModel
   :members:
   
   .. automethod:: __init__

prep_time_series
################

.. module:: utilities
.. autofunction:: prep_time_series

prep_multiple_time_series
#########################
.. module:: utilities
.. autofunction:: prep_multiple_time_series

Additional Help
---------------

If you would like this work extended in areas that are specific to your enterprise, you may submit a request `here <https://www.jonathanbech.tel/contact>`_:
