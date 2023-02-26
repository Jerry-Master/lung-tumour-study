# Soft

This algorithm consists of two U-nets: one for detecting cell nuclei and one for segmentation and classification. After that, the output of both networks is combined using a watershed-esque approach. It was developed by Julia Salas and improved by Feliu Formosa. The code here is just an adaptation of the original one.

## Files

It contains 3 scripts, 2 submodules and 1 external module. The scripts are: `train.py` for training both U-nets, `infer.py` for merging the output of the U-nets and generating predictions, and `model2imgs.py` to visualize the output. The last one is still under construction. The submodules are: `soft_dataset.py` which contains the Dataset class for training both U-nets and `soft_utils.py` which contains auxiliary functions. Lastly, the external module is `soft_segmentation_models_pytorch` which is a modification of `segmentation_models_pytorch` that contains some extra `utils` submodule.
