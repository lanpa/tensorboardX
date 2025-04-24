History
=======
2.6.3 (2025-04-24)
---------------------
* Support PaddlePaddle tensors
* Added a use_metadata context manager
* MoviePy version 2 support

2.6.2.2 (2023-08-20)
---------------------
* Added protobuf's lower bound version (>=3.20)

2.6.2.1 (2023-08-20)
---------------------
* [Yanked because operation miss] Added protobuf's lower bound version (>=3.20)

2.6.2 (2023-07-30)
-------------------
* [Yanked because wrong dependency] Removed version limit for protobuf

2.6.1 (2023-06-18)
-------------------
* Expose use_strict_trace parameter in add_graph (#694)
* Upgrade to protobuf 4
* Fix git based package versioning
* Fix GCS Connection Error #606 (#686)

2.6 (2023-02-12)
-------------------
* Fixed several deprecation warnings
* Update dependencies

2.5.1 (2022-06-05)
-------------------
* Enforce protobuf's version upper bound

2.5 (2022-02-22)
-----------------
* Fix deprecation warnings
* Comet integration improvements (#658)

2.4.1 (2021-11-20)
-------------------
* Fix a comet plugin bug if writer is reused. (#639)

2.4 (2021-06-30)
-----------------
* Remove a dependency issue. (#631)

2.3 (2021-06-20)
-----------------
* Support logging to comet.ml simutaneously.

2.2 (2021-04-03)
-----------------
* Support for type hints.
* Dropped Python 2 support.
* Bug fixes, see the commit log in Github.

2.1 (2020-07-05)
-----------------
* Global SummaryWriter that mimics python's default logger class, concurrent write is supported.
* 200x speed up for add_audio. Please install the soundfile package for this feature.
* Supports jax tensors.
* The add_graph function is delegated to the one in torch.utils.tensorboard.
* Bug fixes, see the commit log in Github.

2.0 (2019-12-31)
-----------------
* Now you can tag Hparams trials with custom name instead of the default epoch time
* Fixed a bug that add_hparams are rendered incorrectly with non-string values
* Supports logging to Amazon S3 or Google Cloud Storage
* Bug fixes and error message for add_embedding function
* Draw openvino format with add_openvino_graph

1.9 (2019-10-04)
-----------------
* Use new JIT backend for pytorch. This works better with pytorch 1.2 and 1.3
* Supports hparams plugin
* add_embedding now supports numpy array input

1.8 (2019-07-05)
-----------------
* Draw label text on image with bounding box provided.
* crc32c speed up (optional by installing crc32c manually)
* Rewrite add_graph. onnx backend is replaced by JIT to support more advanced structure.
* Now you can add_mesh() to visualize colorful point cloud or meshes.

1.7 (2019-05-19)
-----------------
* Able to write to S3
* Fixed raw histogram issue that nothing is shown in TensorBoard
* Users can use various image/video dimension permutation by passing 'dataformats' parameter.
* You can bybass the writer by passing write_to_disk=True to SummaryWriter


1.6 (2019-01-02)
-----------------
* Many graph related bug is fixed in this version.
* New function: add_images(). This function accepts 4D iamge tensor. See documentation.
* Make add_image_with_boxes() usable.
* API change: add_video now accepts BxTxCxHxW instead of BxCxTxHxW tensor.

1.5 (2018-12-10)
-----------------
* Add API for Custom scalar
* Add support for logging directly to S3
* Add support for Caffe2 graph
* Pytorch 1.0.0 JIT graph support (alpha-release)

1.4 (2018-08-09)
-----------------
* Made add_text compatible with tensorboard>1.6
* Fix the issue of strange histogram if default binning method is used
* Supports passing matplotlib figures to add_image()
* Resolve namespace confliction with TF tensorboard
* add_image_boxes function
* Supports custom timestamp for event

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

0.7 (2017-08-22)
-----------------
* remove tensorflow dependency for embedding function
* fixed incorrect image<->label pairing in embedding function (#12)
* unifies API call and adds docstring. Documentation is available at: http://tensorboard-pytorch.readthedocs.io/

0.6.5 (2017-07-30)
------------------
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
