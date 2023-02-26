# lung-tumour-study
[![Python application](https://github.com/Jerry-Master/lung-tumour-study/actions/workflows/python-app.yml/badge.svg)](https://github.com/Jerry-Master/lung-tumour-study/actions/workflows/python-app.yml) [![pyversion](https://raw.githubusercontent.com/Jerry-Master/badges/main/py_version.svg)](https://www.python.org/) [![torchversion](https://raw.githubusercontent.com/Jerry-Master/badges/main/torch_version.svg)](https://pytorch.org/) [![dglversion](https://raw.githubusercontent.com/Jerry-Master/badges/main/dgl_version.svg)](https://www.dgl.ai/)

Python package from my Bachelor's thesis. It is dedicated to aid in the study of tumours.

## Installation

Just do:

```shell
pip install tumourkit
```

It's that simple, dependencies will be automatically installed and several command line scripts will be readily available. It is recommended that you install it inside a virtual environment, for that, type in a shell:

```shell
python -m venv [ENV_NAME]
source [ENV_NAME]/bin/activate
pip install tumourkit
``` 

## Main features

With this package you'll be able to easily convert between different data formats, train models and make inference. As an example, if you want to convert GeoJSON data exported from QuPath into the standar PNG / CSV format you can simply type:

```shell
geojson2pngcsv --gson-dir [...] --png-dir [...] --csv-dir [...]
```

Substituting the dots by the input and output folder.
