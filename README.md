# TumourKit
[![Ubuntu tests](https://github.com/Jerry-Master/lung-tumour-study/actions/workflows/pytest-ubuntu.yml/badge.svg)](https://github.com/Jerry-Master/lung-tumour-study/actions/workflows/pytest-ubuntu.yml) [![Windows tests](https://github.com/Jerry-Master/lung-tumour-study/actions/workflows/pytest-windows.yml/badge.svg)](https://github.com/Jerry-Master/lung-tumour-study/actions/workflows/pytest-windows.yml) [![Documentation Status](https://readthedocs.org/projects/lung-tumour-study/badge/?version=latest)](https://lung-tumour-study.readthedocs.io/en/latest/?badge=latest) [![pyversion](https://raw.githubusercontent.com/Jerry-Master/badges/main/py_versions.svg)](https://www.python.org/) [![torchversion](https://raw.githubusercontent.com/Jerry-Master/badges/main/torch_version.svg)](https://pytorch.org/) [![dglversion](https://raw.githubusercontent.com/Jerry-Master/badges/main/dgl-1.0.1.svg)](https://www.dgl.ai/)

<p align="middle">
  <img src="https://raw.githubusercontent.com/Jerry-Master/lung-tumour-study/main/docs/source/imgs/example.png" width="30%" />
  <img src="https://raw.githubusercontent.com/Jerry-Master/lung-tumour-study/main/docs/source/imgs/overlay.png" width="30%" /> 
  <img src="https://raw.githubusercontent.com/Jerry-Master/lung-tumour-study/main/docs/source/imgs/graph-overlay.png" width="30%" />
</p>

Official implementation of [Combining graph neural networks and computer vision methods for cell nuclei classification in lung tissue](https://doi.org/10.1016/j.heliyon.2024.e28463).

Python package from my Bachelor's thesis. It is dedicated to aid in the study of tumours. It all started with lung tissue, thus the repository name. But it now supports any tissue with any number of classes. For more information, please visit the [official documentation](https://lung-tumour-study.readthedocs.io/en/latest/index.html). Pretrained models can be found on [Hugging Face](https://huggingface.co/Jerry-Master/Hovernet-plus-Graphs).

## Citation

```
@article{PerezCano2024,
  author = {Jose Pérez-Cano and Irene Sansano Valero and David Anglada-Rotger and Oscar Pina and Philippe Salembier and Ferran Marques},
  title = {Combining graph neural networks and computer vision methods for cell nuclei classification in lung tissue},
  journal = {Heliyon},
  year = {2024},
  volume = {10},
  number = {7},
  doi = {10.1016/j.heliyon.2024.e28463},
}
```

