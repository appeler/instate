Instate documentation
=====================

Instate reports how processed occurrences of a surname distribute across
states in the 2017 Indian electoral rolls, as calibrated 0 to 1 proportions,
and derives a language composition by mixing those state shares with Census
2011 mother-tongue shares. The outputs do not establish any person's
residence, origin, language, identity, or behavior.

Results follow the appeler inference contract, composition form: shares sum
to one, unsupported inputs abstain with a machine-readable reason instead of
receiving a default distribution, and provenance columns identify the exact
artifacts used.

Install the package from PyPI:

.. code-block:: console

   pip install instate

Lookup tables ship in the package. The model checkpoint and its calibration
download on first use from the immutable revision configured in
``instate._resources`` and are cached by ``huggingface-hub``. Set
``INSTATE_MODEL_DIR`` to use local artifacts.

State composition
-----------------

``lookup_state_composition`` reports electoral-roll shares for surnames in
the table; ``estimate_state_composition`` runs the temperature-scaled
character BiLSTM for the same quantity, including unseen surnames. Both
preserve every input row and abstain explicitly.

.. code-block:: python

   import instate

   looked_up = instate.lookup_state_composition(["sood", "nair", "unknown123"])
   print(looked_up[["surname", "scored", "abstention_reason",
                    "state_share_punjab", "state_share_kerala"]])

   estimated = instate.estimate_state_composition(["chintalapati"])

Language composition
--------------------

``estimate_language_composition`` mixes state evidence with each state's
Census 2011 mother-tongue shares. The ``basis`` option selects the state
evidence: the electoral lookup, the model, or the default ``auto``, which
prefers the lookup and falls back to the model and records the choice in a
``language_basis`` column.

.. code-block:: python

   languages = instate.estimate_language_composition(["singh", "sharma"])
   print(languages[["surname", "language_basis", "language_share_hindi"]])

Reference lookups
-----------------

.. code-block:: python

   instate.lookup_state_official_languages(["Delhi", "Punjab"])
   instate.list_supported_states()

Reference
---------

.. toctree::
   :maxdepth: 2

   modules

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
