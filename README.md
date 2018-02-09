# PASCAL in Detail API

Python API for the [PASCAL in Detail](https://sites.google.com/view/pasd/dataset) multi-task computer vision challenge. This API is a fork of [MS COCO vision challenge API](https://github.com/pdollar/coco).

## To install:
Run `make` and `make install` under the [PythonAPI](PythonAPI/) directory.

In Python:

```python
from detail import Detail
details = Detail('json/trainval_merged.json', 'VOCdevkit/VOC2010/JPEGImages')
```

If you wish to use edge evaluation, please copy "PythonAPI/detail/benchmark\_ext.so" into where detail package wa installed (for eg. "/usr/local/lib/python2.7/dist-packages/detail").
You can also compile benchmark\_ext yourself. The instruction on how to do that are available in "PythonAPI/detail/benchmark\_ext/README.md".

If you wish to use the API from MATLAB, see [MATLAB's documentation for calling Python code](https://www.mathworks.com/help/matlab/matlab_external/call-python-from-matlab.html). The Detail API no longer maintains a separate MATLAB API.

## To see a demo:

Run the IPython notebook [PythonAPI/ipynb/detailDemo.ipynb](PythonAPI/ipynb/detailDemo.ipynb).
