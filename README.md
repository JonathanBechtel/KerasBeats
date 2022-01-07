# KerasBeats
----
An easy, accessible way to use the NBeats model architecture in Keras.

![kerasbeats](common/images/nbeats.PNG "N-Beats Model Architecture")

 **Table of Contents:**
   - [Introduction](###Introduction)
   - [Installation](###Installation)
   - [Basic Usage](###Basic%20Usage)
   - [Data Prep](###Data%20Prep)
   - [KerasBeats layer](###KerasBeats%20layer)
   - [KerasBeats as keras model](###KerasBeats%20as%20keras%20model)

### Introduction
The motivation for this project was to take the NBeats model architecture defined in the original paper here:  https://arxiv.org/abs/1905.10437 and reproduce it in a widely accessible form in keras.  In the past few years this model has become very popular as a timer series forecasting tool, but its implementation in keras seemed elusive, without an easy-to-use, well documented option online that'd be simple for newcomers to try. When you are looking to use a new tool for the first time, it's vital that you can do a simple install and quickly use an api syntax you are already familiar with to get started within minutes.  

To that end, KerasBeats was built with the following ideas in mind:
 - It should reflect the original model architecture as closely as possible.
 - It should have a simple, high level architecture that allows people to get started as quickly as possible using the familar `fit/predict` methods that everyone is already familiar with
 - It should allow you to quickly and easily use it as a keras model to take advantage of the libraries existing functionality and enable its use in existing workflows
 
### Installation
kerasbeats can be installed with the following line: 

```pip install keras-beats```

### Basic Usage

The N-Beats model architecture assumes that you take a univariate time series and create training data that contains previous values for an observation at a particular point in time.  For example, let's assume you have the followin very simple univariate time series:

```
# sample time series values
time_vals = [1, 2, 3, 4, 5]
```

If you were predicting one period ahead and wanted to use the previous two values in the time series as input, you want your data to be formatted like this:

```
# data formatting for N-beats
# each row represents the previous two values for the currently observed one
X = [[1, 2],
     [2, 3],
     [3, 4]]
     
y = [[3, 4, 5]
]```

The idea here is that `[1, 2]` were the two values that preceded `3`, `[2, 3]` were the two that preceeded `4`, and so on.  

Once your input data is formatted like this then you can use `kerasbeats` in the followin way:

```
from kerasbeats import NBeatsModel
mod = NBeatsModel()
mod.fit(X, y)
```

When you are finished fitting your model you can use the `predict` and `evaluate` methods, which are just wrappers on the original keras methods, and would work in exactly the same way.

### Data Prep
Most time series data typically comes in column format, so a little data prep is usually needed before you can feed it into `kerasbeats`, you can easily do this yourself, but there are some built in functions in the `kerasbeats` package to make this a little easier.  

If you have a single time series, you can use the `prep_time_series` function to get your data in the appropriate format.  It works like this:

```from kerasbeats import prep_time_series
# sample data:  a mock time series with twenty values
time_vals = np.arange(10)
windows, labels = prep_time_series(lookback = 5, horizon = 1)```

Once you are done with this the value of `windows` will be the following numpy array:

```
array([[0, 1, 2, 3, 4],
       [1, 2, 3, 4, 5],
       [2, 3, 4, 5, 6],
       [3, 4, 5, 6, 7],
       [4, 5, 6, 7, 8]])
 ```
       
The value of `labels` will be the following numpy array:

```
array([[5],
       [6],
       [7],
       [8],
       [9]])
 ```
     
### KerasBeats layer
Data goes here.

### KerasBeats as keras model
Data goes here.
