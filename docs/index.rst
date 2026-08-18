Instate documentation
=====================

Instate reports how processed surname occurrences are distributed across the
included state records in 2017 electoral rolls. Its character-level BiLSTM
models rank that state target and a synthetic mixture of ranked state
languages. The outputs do not establish any person's residence, origin,
language, identity, or behavior.

Install the package from PyPI:

.. code-block:: console

   pip install instate

Lookup tables ship in the package. Neural checkpoints are downloaded on first
use from the immutable revision configured in ``instate._resources`` and cached
by ``huggingface-hub``. Set ``INSTATE_MODEL_DIR`` to use local checkpoints.

Electoral-roll lookup
---------------------

``get_state_distribution`` preserves every input row, including duplicates,
short names, missing values, and names absent from the lookup table. Unmatched
rows have missing state shares.

.. code-block:: python

   import instate

   result = instate.get_state_distribution(["sood", "nair", "unknown"])
   print(result[["name", "Punjab", "Tamil Nadu"]])

Model prediction
----------------

``predict_state`` and the LSTM form of ``predict_language`` return the requested
number of ranked labels for names with at least three supported ASCII letters.
Short or unsupported names receive an empty list and an explicit
``prediction_status`` reason. ``get_model_metadata`` describes the supported
input for each model path. Neural rankings are based on uncalibrated raw model
scores, not probabilities.

.. code-block:: python

   states = instate.predict_state(["kumar", "patel"], top_k=3)
   languages = instate.predict_language(["singh", "sharma"], top_k=3)
   print(states[["predicted_states", "prediction_status"]])

The KNN language lookup returns one language per name:

.. code-block:: python

   languages = instate.predict_language(["singh"], model="knn")

Reference
---------

.. toctree::
   :maxdepth: 2

   modules

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
