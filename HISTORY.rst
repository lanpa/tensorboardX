History
=======
1.2 (2018-04-21)
-----------------
* Supports tensorshape information in graph visualization. Drop support for 0.3.1
* Adds add_video function

1.1 (2018-02-21)
-----------------
* Supports pytorch 0.3.1 (hacky)

1.0 (2018-01-18)
-----------------
* Supports graph (the pretty one)

0.9 (2017-11-11)
-----------------
* Supports markdown for add_text function
* It's ready to log precision recall curve (needs tensorboard>=0.4)
* Adds context manager for the SummaryWriter class

0.8 (2017-09-25)
-----------------
* Package name renamed to tensorboardX to fix namespace confliction with tensorflow's tensorboard
* Supports multi-scalars and JSON export
* Multiple Embeddings in One Experiment 
* Supports Chainer and mxnet

* fixed incorrect image<->label pairing in embedding function (#12)
* unifies API call and adds docstring. Documentation is available at: http://tensorboard-pytorch.readthedocs.io/

0.7 (2017-08-22)
-----------------
* remove tensorflow dependency for embedding function
* fixed incorrect image<->label pairing in embedding function (#12)
* unifies API call and adds docstring. Documentation is available at: http://tensorboard-pytorch.readthedocs.io/

0.6.5 (2017-07-30)
-----------------
* add travis test (py2.7, py3.6)
* add support for python2 (in PyPI)

0.6 (2017-07-18)
-----------------
* supports embedding

0.5 (2017-07-18)
-----------------
* supports graph summary
* fixed np.histogram issue

0.4 (2017-07-12)
-----------------
* supports text summary

0.3 (2017-07-03)
-----------------
* supports audio summary

0.2 (2017-06-24)
-----------------
* simplifies add_image API
* speed up add_histogram API by 35x


0.1 (2017-06-13)
------------------
* First commit. Reference:
https://github.com/TeamHG-Memex/tensorboard_logger
https://github.com/dmlc/tensorboard
