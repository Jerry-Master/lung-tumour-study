# TumourKit
[![pyversion](https://raw.githubusercontent.com/Jerry-Master/badges/main/py_version.svg)](https://www.python.org/) [![torchversion](https://raw.githubusercontent.com/Jerry-Master/badges/main/torch_version.svg)](https://pytorch.org/) [![dglversion](https://raw.githubusercontent.com/Jerry-Master/badges/main/dgl-1.0.1.svg)](https://www.dgl.ai/)

Python package from my Bachelor's thesis. It is dedicated to aid in the study of tumours.

## Installation

Just do:

```shell
pip install tumourkit
```

It's that simple, dependencies will be automatically installed and several command line scripts will be readily available. It is recommended that you install it inside a virtual environment. For that, type in a shell:

```shell
python3.10 -m venv [ENV_NAME]
source [ENV_NAME]/bin/activate
pip install tumourkit
```

PyTorch and Deep Graph Library are not included as dependencies since they change rapidly and require extra links. To install PyTorch and Deep Graph Library please go to their official installation pages. 

### Known errors

If you try to use the GPU version of Deep Graph Library but have installed the CPU you will receive the following error.

```shell
Check failed: allow_missing: Device API gpu is not enabled. Please install the cuda version of dgl.
```

Just install the GPU build and you will be fine.

If you come accross something like

```shell
OSError: libcusparse.so.11: cannot open shared object file: No such file or directory
```

That means your python environment does not link correctly to your CUDA installation. You will have to edit the `LD_LIBRARY_PATH` environmental variable so that the dynamic library `libcusparse.so.11` can be found. Typically it is found under `nvidia/cublas/lib` so a possible fix is

```shell
export LD_LIBRARY_PATH=[ENV_NAME]/lib/python3.10/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH
```

## Main features

With this package you'll be able to easily convert between different data formats, train models and make inference. As an example, if you want to convert GeoJSON data exported from QuPath into the standard PNG / CSV format you can simply type:

```shell
geojson2pngcsv --gson-dir [...] --png-dir [...] --csv-dir [...]
```

Substituting the dots by the input and output folders respectively.
