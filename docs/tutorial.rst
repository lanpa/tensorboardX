Tutorials
*********

What is tensorboard X?
----------------------

At first, the package was named tensorboard, and soon there are issues about
name confliction. The first alternative name came to my mind is
tensorboard-pytorch, but in order to make it more general, I chose tensorboardX
which stands for tensorboard for X.

Google's tensorflow's tensorboard is a web server to serve visualizations of the
training progress of a neural network, it visualizes scalar values, images,
text, etc.; these information are saved as events in tensorflow. It's a pity
that other deep learning frameworks lack of such tool, so there are already
packages letting users to log the events without tensorflow; however they only
provides basic functionalities. The purpose of this package is to let
researchers use a simple interface to log events within PyTorch (and then show
visualization in tensorboard). This package currently supports logging scalar,
image, audio, histogram, text, embedding, and the route of back-propagation. The
following manual is tested on Ubuntu and Mac, and the environment are anaconda's
python2 and python3.


create a summary writer
-----------------------
Before logging anything, we need to create a writer instance. This can be done with:

.. code-block:: python

    from tensorboard import SummaryWriter
    #SummaryWriter encapsulates everything
    writer = SummaryWriter('runs/exp-1')
    #creates writer object. The log will be saved in 'runs/exp-1'
    writer2 = SummaryWriter()
    #creates writer2 object with auto generated file name, the dir will be something like 'runs/Aug20-17-20-33'
    writer3 = SummaryWriter(comment='3x learning rate')
    #creates writer2 object with auto generated file name, the comment will be appended to the filename. The dir will be something like 'runs/Aug20-17-20-33-3xlearning rate'

Each subfolder will be treated as different experiments in tensorboard. Each
time you re-run the experiment with different settings, you should change the
name of the sub folder such as ``runs/exp2``, ``runs/myexp`` so that you can
easily compare different experiment settings. Type ``tensorboard runs`` to compare
different runs in tensorboard.


general api format
------------------
.. code-block:: python

    add_something(tag name, object, iteration number)


add scalar
-----------
Scalar value is the most simple data type to deal with. Mostly we save the loss
value of each training step, or the accuracy after each epoch. Sometimes I save
the corresponding learning rate as well. It's cheap to save scalar value. Just
log anything you think is important. To log a scalar value, use
``writer.add_scalar('myscalar', value, iteration)``. Note that the program complains
if you feed a PyTorch variable. Remember to extract the scalar value by
``x.data[0]`` if ``x`` is a torch variable.


add scalars
-----------



add image
---------
An image is represented as 3-dimensional tensor. The simplest case is save one
image at a time. In this case, the image should be passed as a 3-dimension
tensor of size ``[3, H, W]``. The three dimensions correspond to R, G, B channel of
an image. After your image is computed, use ``writer.add_image('imresult', x,
iteration)`` to save the image. If you have a batch of images to show, use
``torchvision``'s ``make_grid`` function to prepare the image array and send the result
to ``add_image(...)`` (``make_grid`` takes a 4D tensor and returns tiled images in 3D tensor)

.. Note::
	Remember to normalize your image.


add histogram
-------------
Saving histogram is expensive. Both in computation time and storage. If training
slows down after using this package, check this first. To save a histogram,
convert the array into numpy array and save with ``writer.add_histogram('hist',
array, iteration)``.

add video
---------


add figure
----------


add text
--------


add prcurve
-----------

add graph
---------
Graph drawing is based on ``autograd``'s backward tracing. It goes along the
next_functions attribute in a variable recursively, drawing each encountered
nodes. To draw the graph, you need a model ``m`` and an input variable ``t``
that have correct size for ``m``. Let ``r = m(t)``, then please invoke
``writer.add_graph(m, r)`` to save the graph. By default, the input tensor does not
require gradient, therefore it will be omitted when back tracing. To draw the
input node, pass an additional parameter ``requires_grad=True`` when creating the
input tensor. See
`The graph demo <https://github.com/lanpa/tensorboardX/blob/master/examples/demo_graph.py>`_ for
complete example.


add audio
---------
Currently the sampling rate of the this function is fixed at 44100 KHz, single
channel. The input of the add_audio function is a one dimensional array, with
each element representing the consecutive amplitude samples. For a 2 seconds
audio, the input ``x`` should have 88200 elements. Each element should lie in
[-1, 1].

add embedding
-------------
what is embedding?
==================


visualization
=============
Embedding is a technique to visualize high dimensional data. To convert high
dimensional data into human perceptible 3D data, tensorboard provides PCA and
t-sne to project the data into low dimensional space. What you need to do is
provide a bunch of points and tensorboard will do the rest for you. The bunch of
points is passed as a tensor of size ``n x d``, where ``n`` is the number of points and
``d`` is the feature dimension. The feature representation can either be raw data
(e.g. the MNIST image) or a representation learned by your network (extracted
feature). This determines how the points distributes. To make the visualization
more informative, you can pass optional metadata or ``label_imgs`` for each data
points. In this way you can see that neighboring point have similar label and
distant points have very different label (semantically or visually). Here the
metadata is a list of labels, and the length of the list should equal to n, the
number of the points. The label_imgs is a 4D tensor of size ``NCHW``. ``N`` should equal
to ``n`` as well. See
`The embedding demo <https://github.com/lanpa/tensorboardX/blob/master/demo_embedding.py>`_ for
complete example.


useful commands
---------------
install
=======

Simply type ``pip install tensorboardX`` in Bash to install this package.
To use the newest version, you might need to build from source or ``pip install
tensorboardX —-no-cache-dir`` .  To run tensorboard web server, you need
to install tensorflow by ``pip install tensorflow`` or ``pip install tensorflow-gpu``.
After that, type ``tensorboard --logdir=<yourlogdir>`` to start the server, where
``yourlogdir`` is the parameter of the object constructor. I think this command is
tedious, so I add a line alias ``tb='tensorboard --logdir '`` in ``~/.bash_profile``. In
this way, the above command is simplified as ``tb <yourlogdir>``. Use your favorite
browser to load the tensorboard page, the address will be shown in the terminal
after starting the server.



run tensorboard server
======================

show more images in tensorboard
===============================



misc
----


performance issue
=================
Logging is cheap, but display is expensive.
For my experience, if there are 3 or more experiments to show at a time and each
experiment have, say, 50K points, tensorboard might need a lot of time to
present the data.


Grouping plots
==============
Usually, there are many numbers to log in one experiment. For example, when
training GANs you should log the loss of the generator, discriminator. If the
loss is composed of two other loss functions, say L1 and MSE, you might want to
log the value of the other two losses as well. In this case, you can write the
tags as Gen/L1, Gen/MSE, Desc/L1, Desc/MSE. In this way, tensorboard will group
the plots into two sections (Gen, Desc). You can also use the regular expression
to filter data.
