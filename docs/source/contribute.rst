Contribute
==========

You can support the project by adding graph models or increasing the number of features used.
Every contribution must have type hints and proper docstrings.

.. _gnn:

Expanding the graph module
--------------------------

Models
^^^^^^

In case you are interested in adding more models to the graph zoo, you can do so modifying the source code and creating a pull request. 
Models should go in the folder :code:`tumourkit/classification/models` and must adhere to the following API:

.. code-block:: python

  class NewGraph(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers, drop_rate, norm_type):
        super(NewGraph, self).__init__()
        # Add any layers you want here
        # ...
        # Dropout layers should be defined as
        # nn.Dropout(drop_rate)
        # Normalization layers should be defined a
        # Norm(norm_type=norm_type, hidden_dim=h_feats)

    def forward(self, g, in_feat):
        h = in_feat
        # Make any computation with the hidden embedding h here
        # ...
        return h

Your model should have a variable amount of layers controlled by :code:`num_layers`, a variable amount of dropout rate as given by :code:`drop_rate` 
and must include either batch normalization or no normalization, both controlled through :py:func:`Norm <tumourkit.classification.models.norm.Norm>`.

Extracted features
^^^^^^^^^^^^^^^^^^

Currently, the features that are used by the models are:

* The area and perimeter of the cell, in pixels.
* The standard deviation of the pixel values in gray format.
* The histogram of the red, green and blue channels quantized into five bins each.
* The prior probability of the class as given by Hovernet.

If you want to add more features, you should change two files: :code:`tumourkit/classification/train_graphs.py`, :code:`tumourkit/utils/preprocessing.py`. 
In the first one you need to modify the :py:func:`load_model <tumourkit.classification.train_graphs.load_model>` function.

.. code-block:: python

  def load_model(conf: Dict[str,Any], num_classes: int) -> nn.Module:
      # ...
      num_feats = 18 + (1 if num_classes == 2 else num_classes)
      # ...

In there modify the :code:`num_feats` variable to denote the number of features there are. If you add 2 more features, change the 18 by a 20.

The other function needed to be adapted is :py:func:`extract_features <tumourkit.utils.preprocessing.extract_features>`:

.. code-block:: python

  def extract_features(msk_img: np.ndarray, bin_msk: np.ndarray, debug=False) -> Dict[str, np.ndarray]:
      # ...
      feats = {}
      # Add any feature you want to the dictionary feats.
      # You can use the keys you want, they will be reflected as columns in the saved files.
      return feats