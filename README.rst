instate: predict spoken language and the state of residence from last name 
=============================================================

.. image:: https://img.shields.io/pypi/v/instate.svg
    :target: https://pypi.python.org/pypi/instate
.. image:: https://readthedocs.org/projects/instate/badge/?version=latest
    :target: http://instate.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://static.pepy.tech/badge/instate
    :target: https://pepy.tech/project/instate


Using the Indian electoral rolls data (2017), we provide a Python package that takes the last name of a person and gives its distribution across states.
This package can also predict the spoken language of the person based on the last name.

Potential Use Cases
---------------------
India has 22 official languages. To serve such a diverse language base is a challenge for businesses and surveyors. To the extent that businesses have access to the last name (and no other information) and in the absence of other data that allows us to model a person's spoken language, the distribution of last names across states is the best we have.

Dataset
---------
Refer `lastname_langs_india.csv.tar.gz <https://github.com/appeler/instate/blob/main/instate/data/lastname_langs_india.csv.tar.gz>`__ for the dataset, that will be used to predict/lookup the spoken language based on the last name.

Refer `lastname_langs_india_top3.csv.tar.gz <https://github.com/appeler/instate/blob/main/instate/data/lastname_langs_india_top3.csv.tar.gz>`__ for the dataset, that will be used to predict the top-3 spoken languages based on the last name. A LSTM model has been trained on this dataset to predict the top-3 spoken languages.

Refer `notebooks <https://github.com/appeler/instate/tree/main/instate/notebooks>`__ for the notebooks that were used to prepare above datasets and train the models.

Web UI
--------------
Streamlit App.: https://appeler-instate-streamlitstreamlit-app-e39m4c.streamlit.app/

Installation
-------------
We strongly recommend installing `instate` inside a Python virtual environment
(see `venv documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`__)

::

    pip install instate

Examples
--------
::

  from instate import last_state
  last_dat <- pd.read_csv("last_dat.csv")
  last_state_dat <- last_state(last_dat, "dhingra")
  print(last_state_dat)

API
----------

instate exposes 5 functions. 

- **last_state**

    - takes a pandas dataframe, the column name for the df column with the last names, and produces a dataframe with 31 more columns, reflecting the number of states for which we have the data. 

::
    
    from instate import last_state
    df = pd.DataFrame({'last_name': ['Dhingra', 'Sood', 'Gowda']})
    last_state(df, "last_name").iloc[:, : 5]
        
        last_name   __last_name andaman     andhra      arunachal
    0   Dhingra     dhingra     0.001737    0.000744    0.000000
    1   Sood        sood        0.000258    0.002492    0.000043
    2   Gowda       gowda       0.000000    0.528533    0.000000

- **pred_last_state**
    
    - takes a pandas dataframe, the column name with the last names, and produces a dataframe with 1 more column (pred_state), reflecting the top-3 predictions from GRU model.

::
    
    from instate import pred_last_state
    df = pd.DataFrame({'last_name': ['Dhingra', 'Sood', 'Gowda']})
    last_state(df, "last_name").iloc[:, : 5]
        last_name	pred_state
    0	dhingra	[Daman and Diu, Andaman and Nicobar Islands, Puducherry]
    1	sood	[Meghalaya, Chandigarh, Punjab]
    2	gowda	[Puducherry, Nagaland, Daman and Diu]

- **state_to_lang**

    - takes a pandas dataframe, the column name with the state, and appends census mappings from state to languages

::

  from instate import state_to_lang
  df = pd.DataFrame({'last_name': ['dhingra', 'sood', 'gowda']})
  state_last = last_state(df, "last_name")
  small_state = state_last.loc[:, "andaman":"utt"]
  state_last["modal_state"] = small_state.idxmax(axis = 1)
  state_to_lang(state_last, "modal_state")[["last_name", "modal_state", "official_languages"]]

        last_name   modal_state official_languages
    0   dhingra     delhi       Hindi, English
    1   sood        punjab      Punjabi
    2   gowda       andhra      Telugu


- **lookup_lang**

    - takes a pandas dataframe, the column name with the last names, and produces a dataframe with 1 more column (lang), reflecting the most spoken language in the state. This method will find nearest names and then look up in dataset to find the most spoken language.

::
    
      from instate import lookup_lang
      df = pd.DataFrame({'last_name': ['sood', 'chintalapati']})
      lookup_lang(df, "last_name")
      
            last_name predicted_lang
    0          sood          hindi
    1  chintalapati         telugu

- **predict_lang**

    - takes a pandas dataframe, the column name with the last names, and produces a dataframe with 1 more column (lang), reflecting the most spoken language in the state. This method will predict the language based on the names.

::
    
      from instate import predict_lang
      df = pd.DataFrame({'last_name': ['sood', 'chintalapati']})
      predict_lang(df, "last_name")
      
            last_name predicted_lang
    0          sood   [hindi, punjabi, urdu]
    1  chintalapati  [telugu, urdu, chenchu]

Data
----

The underlying data for the package can be accessed at: https://doi.org/10.7910/DVN/ZXMVTJ

Evaluation
----------

The model has a top-3 accuracy of 85.3\% on `unseen names <https://github.com/appeler/instate/blob/main/instate/models/model_dnn_gpu.ipynb>`__. The KNN model does quite well. See the details `here <https://github.com/appeler/instate/blob/main/instate/models/KNN_cosine_distance_simple_avg_modal_state.ipynb>`__
The name-to-language lookup has an accuracy of 67.9\%.
The name-to-language model prediction has an accuracy of 72.2\%.

Authors
-------

Atul Dhingra, Gaurav Sood and Rajashekar Chintalapati

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
