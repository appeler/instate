==================================================
instate: predict the state of residence from last name 
==================================================

.. image:: https://app.travis-ci.com/appeler/instate.svg?branch=master
    :target: https://travis-ci.org/appeler/instate
.. image:: https://ci.appveyor.com/api/projects/status/5wkr850yy3f6sg6a?svg=true
    :target: https://ci.appveyor.com/project/soodoku/instate
.. image:: https://img.shields.io/pypi/v/instate.svg
    :target: https://pypi.python.org/pypi/instate
.. image:: https://readthedocs.org/projects/instate/badge/?version=latest
    :target: http://notnews.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://pepy.tech/badge/instate
    :target: https://pepy.tech/project/instate


Using the Indian electoral rolls data (2017), we provide a Python package that takes the last name of a person and gives its distribution across states. 

Potential Use Cases
---------------------
India has 22 official languages. And to serve such a diverse language base is a challenge for businesses and surveyors. To the extent that businesses have access to the last name (and no other information) and in absence of other data that allows us to model a person's spoken language, the distribution of last name across states is the best we have.

Installation
-------------
We strongly recommend installing `indicate` inside a Python virtual environment
(see `venv documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`__)

::

    pip install instate

General API
-----------
1. transliterate.hindi2english will take Hindi text and translate into English.

Examples
--------
::

  from instate import last_state
  last_dat <- pd.read_csv("last_dat.csv")
  last_state_dat <- last_state(last_dat, "dhingra")
  print(last_state_dat)

output -


API
----------

instate exposes 3 functions. 

- **last_state**

  - What it does:

    - takes a pandas dataframe, the column name with the last names, and produces a dataframe with XX more columns, reflecting the number of states for which we have the data. 

  - Output

- **pred_last_state**
    
  - What it does:

    - takes a pandas dataframe, the column name with the last names, and produces a dataframe with XX more columns, reflecting the number of states for which we have the data. 

  - Output

- **state_to_lang**
    - 

Data
----

The underlying data for the package can be accessed at: https://doi.org/10.7910/DVN/ENXOJE

Evaluation
----------

Authors
-------

Atul Dhingra and Gaurav Sood

Contributor Code of Conduct
---------------------------------

The project welcomes contributions from everyone! In fact, it depends on
it. To maintain this welcoming atmosphere, and to collaborate in a fun
and productive way, we expect contributors to the project to abide by
the `Contributor Code of
Conduct <http://contributor-covenant.org/version/1/0/0/>`__.

License
----------

The package is released under the `MIT
License <https://opensource.org/licenses/MIT>`__.