.. instate documentation master file, created by
   sphinx-quickstart on Tue Mar 14 18:24:40 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to instate's documentation!
===================================


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

About instate
=============

instate is a Python package that predicts spoken language and state of residence from last names using Indian electoral rolls data (2017) and neural network models.

The package provides two main approaches:

**Electoral Rolls Lookups** - Fast frequency-based lookups for names found in Indian electoral rolls data:
  - :func:`instate.get_state_distribution` - Get P(state|lastname) from electoral rolls
  - :func:`instate.get_state_languages` - Get languages spoken in predicted states

**Neural Network Predictions** - For names not in electoral rolls or for enhanced predictions:
  - :func:`instate.predict_state` - Predict states using GRU neural networks  
  - :func:`instate.predict_language` - Predict languages using LSTM or KNN

Quick Start
-----------

.. code-block:: python

   import instate
   
   # Get state distribution from electoral rolls
   result = instate.get_state_distribution(['sood', 'dhingra'])
   
   # Predict states with neural networks
   predictions = instate.predict_state(['kumar', 'patel'], top_k=3)
   
   # Predict languages
   languages = instate.predict_language(['singh', 'sharma'], model='lstm')

For complete documentation and examples, please see the `GitHub README <https://github.com/appeler/instate>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`