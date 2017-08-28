# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides an API for generating Event protocol buffers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import json
import os
from .src import event_pb2
from .src import summary_pb2
from .src import graph_pb2
from .event_file_writer import EventFileWriter
from .summary import scalar, histogram, image, audio, text
from .graph import graph
from .embedding import make_mat, make_sprite, make_tsv, append_pbtxt


class SummaryToEventTransformer(object):
    """Abstractly implements the SummaryWriter API.
    This API basically implements a number of endpoints (add_summary,
    add_session_log, etc). The endpoints all generate an event protobuf, which is
    passed to the contained event_writer.
    @@__init__
    @@add_summary
    @@add_session_log
    @@add_graph
    @@add_meta_graph
    @@add_run_metadata
    """

    def __init__(self, event_writer, graph=None, graph_def=None):
        """Creates a `SummaryWriter` and an event file.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers constructed when you
        call one of the following functions: `add_summary()`, `add_session_log()`,
        `add_event()`, or `add_graph()`.
        If you pass a `Graph` to the constructor it is added to
        the event file. (This is equivalent to calling `add_graph()` later).
        TensorBoard will pick the graph from the file and display it graphically so
        you can interactively explore the graph you built. You will usually pass
        the graph from the session in which you launched it:
        ```python
        ...create a graph...
        # Launch the graph in a session.
        sess = tf.Session()
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.summary.FileWriter(<some-directory>, sess.graph)
        ```
        Args:
          event_writer: An EventWriter. Implements add_event method.
          graph: A `Graph` object, such as `sess.graph`.
          graph_def: DEPRECATED: Use the `graph` argument instead.
        """
        self.event_writer = event_writer
        # For storing used tags for session.run() outputs.
        self._session_run_tags = {}
        # TODO(zihaolucky). pass this an empty graph to check whether it's necessary.
        # currently we don't support graph in MXNet using tensorboard.

    def add_summary(self, summary, global_step=None):
        """Adds a `Summary` protocol buffer to the event file.
        This method wraps the provided summary in an `Event` protocol buffer
        and adds it to the event file.
        You can pass the result of evaluating any summary op, using
        [`Session.run()`](client.md#Session.run) or
        [`Tensor.eval()`](framework.md#Tensor.eval), to this
        function. Alternatively, you can pass a `tf.Summary` protocol
        buffer that you populate with your own data. The latter is
        commonly done to report evaluation results in event files.
        Args:
          summary: A `Summary` protocol buffer, optionally serialized as a string.
          global_step: Number. Optional global step value to record with the
            summary.
        """
        if isinstance(summary, bytes):
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ
        event = event_pb2.Event(summary=summary)
        self._add_event(event, global_step)

    def add_graph(self, graph):
        """Adds a `Graph` protocol buffer to the event file.
        """
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self._add_event(event, None)

    def add_session_log(self, session_log, global_step=None):
        """Adds a `SessionLog` protocol buffer to the event file.
        This method wraps the provided session in an `Event` protocol buffer
        and adds it to the event file.
        Args:
          session_log: A `SessionLog` protocol buffer.
          global_step: Number. Optional global step value to record with the
            summary.
        """
        event = event_pb2.Event(session_log=session_log)
        self._add_event(event, global_step)

    def _add_event(self, event, step):
        event.wall_time = time.time()
        if step is not None:
            event.step = int(step)
        self.event_writer.add_event(event)


class FileWriter(SummaryToEventTransformer):
    """Writes `Summary` protocol buffers to event files.
    The `FileWriter` class provides a mechanism to create an event file in a
    given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    @@__init__
    @@add_summary
    @@add_session_log
    @@add_event
    @@add_graph
    @@add_run_metadata
    @@get_logdir
    @@flush
    @@close
    """

    def __init__(self,
                 logdir,
                 graph=None,
                 max_queue=10,
                 flush_secs=120,
                 graph_def=None):
        """Creates a `FileWriter` and an event file.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers constructed when you
        call one of the following functions: `add_summary()`, `add_session_log()`,
        `add_event()`, or `add_graph()`.
        If you pass a `Graph` to the constructor it is added to
        the event file. (This is equivalent to calling `add_graph()` later).
        TensorBoard will pick the graph from the file and display it graphically so
        you can interactively explore the graph you built. You will usually pass
        the graph from the session in which you launched it:
        ```python
        ...create a graph...
        # Launch the graph in a session.
        sess = tf.Session()
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.summary.FileWriter(<some-directory>, sess.graph)
        ```
        The other arguments to the constructor control the asynchronous writes to
        the event file:
        *  `flush_secs`: How often, in seconds, to flush the added summaries
           and events to disk.
        *  `max_queue`: Maximum number of summaries or events pending to be
           written to disk before one of the 'add' calls block.
        Args:
          logdir: A string. Directory where event file will be written.
          graph: A `Graph` object, such as `sess.graph`.
          max_queue: Integer. Size of the queue for pending events and summaries.
          flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk.
          graph_def: DEPRECATED: Use the `graph` argument instead.
        """
        event_writer = EventFileWriter(logdir, max_queue, flush_secs)
        super(FileWriter, self).__init__(event_writer, graph, graph_def)

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self.event_writer.get_logdir()

    def add_event(self, event):
        """Adds an event to the event file.
        Args:
          event: An `Event` protocol buffer.
        """
        self.event_writer.add_event(event)

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to
        disk.
        """
        self.event_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.
        Call this method when you do not need the summary writer anymore.
        """
        self.event_writer.close()

    def reopen(self):
        """Reopens the EventFileWriter.
        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file.
        Does nothing if the EventFileWriter was not closed.
        """
        self.event_writer.reopen()


class SummaryWriter(object):
    """Writes `Summary` directly to event files.
    The `SummaryWriter` class provides a high-level api to create an event file in a
    given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """
    def __init__(self, log_dir=None, comment=''):
        """

        Args:
            log_dir (string): save location, default is: runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run. Use hierarchical folder structure to compare between runs easily. e.g. 'runs/exp1', 'runs/exp2'
            comment (string): comment that appends to the default log_dir
        """
        if log_dir == None:
            import socket
            from datetime import datetime
            log_dir = os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname()+comment)
        self.file_writer = FileWriter(logdir=log_dir)
        v = 1E-12
        buckets = []
        neg_buckets = []
        while v < 1E20:
            buckets.append(v)
            neg_buckets.append(-v)
            v *= 1.1
        self.default_bins = neg_buckets[::-1] + [0] + buckets
        self.text_tags = []
    def add_scalar(self, tag, scalar_value, global_step=None):
        """Add scalar data to summary.

        Args:
            tag (string): Data identifier
            scalar_value (float): Value to save
            global_step (int): Global step value to record

        """
        self.file_writer.add_summary(scalar(tag, scalar_value), global_step)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow'):
        """Add histogram to summary.

        Args:
            tag (string): Data identifier
            values (numpy.array): Values to build histogram
            global_step (int): Global step value to record
            bins (string): one of {'tensorflow','auto', 'fd', ...}, this determines how the bins are made. You can find other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html

        """
        if bins=='tensorflow':
            bins = self.default_bins
        self.file_writer.add_summary(histogram(tag, values, bins), global_step)

    def add_image(self, tag, img_tensor, global_step=None):
        """Add image data to summary.

        Args:
            tag (string): Data identifier
            img_tensor (torch.Tensor): Image data
            global_step (int): Global step value to record
        Shape:
            img_tensor: :math:`(3, H, W)`. Use ``torchvision.utils.make_grid()`` to prepare it is a good idea.
        """
        self.file_writer.add_summary(image(tag, img_tensor), global_step)
    def add_audio(self, tag, snd_tensor, global_step=None):
        """Add audio data to summary.

        Args:
            tag (string): Data identifier
            snd_tensor (torch.Tensor): Sound data
            global_step (int): Global step value to record

        Shape:
            snd_tensor: :math:`(1, L)`. The values should between [-1, 1]. The sample rate is currently fixed at 44100 KHz.
        """
        self.file_writer.add_summary(audio(tag, snd_tensor), global_step)
    def add_text(self, tag, text_string, global_step=None):
        """Add text data to summary.

        Args:
            tag (string): Data identifier
            text_string (string): String to save
            global_step (int): Global step value to record

        Examples::

            writer.add_text('lstm', 'This is an lstm', 0)
            writer.add_text('rnn', 'This is an rnn', 10)

        """        
        self.file_writer.add_summary(text(tag, text_string), global_step)
        if tag not in self.text_tags:
            self.text_tags.append(tag)
            extensionDIR = self.file_writer.get_logdir()+'/plugins/tensorboard_text/'
            if not os.path.exists(extensionDIR):
                os.makedirs(extensionDIR)
            with open(extensionDIR + 'tensors.json', 'w') as fp:
                json.dump(self.text_tags, fp)
    def add_graph(self, model, lastVar):
        # prohibit second call?
        # no, let tensorboard handles it and show its warning message.
        """Add graph data to summary.

            To draw the graph, you need a model ``m`` and an input variable ``t`` that have correct size for ``m``. 
            Say you have runned ``r = m(t)``, then you can use ``writer.add_graph(m, r)`` to save the graph. 
            By default, the input tensor does not require gradient, therefore it will be omitted when back tracing. 
            To draw the input node, pass an additional parameter ``requires_grad=True`` when creating the input tensor.

        Args:
            model (torch.nn.Module): model to draw. 
            lastVar (torch.autograd.Variable): the root node start from.

        .. note::
            This is experimental feature. Graph drawing is based on autograd's backward tracing. 
            It goes along the ``next_functions`` attribute in a variable recursively, drawing each encountered nodes. 
            In some cases, the result is strange. See  https://github.com/lanpa/tensorboard-pytorch/issues/7 and https://github.com/lanpa/tensorboard-pytorch/issues/9
        """      
        import torch
        if not hasattr(torch.autograd.Variable, 'grad_fn'):
            print('pytorch version is too old, how about build by yourself?')
            return
        self.file_writer.add_graph(graph(model, lastVar))

    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None):
        """Add embedding projector data to summary.

        Args:
            mat (torch.Tensor): A matrix which each row is the feature vector of the data point
            metadata (list): A list of labels, each element will be convert to string
            label_img (torch.Tensor): Images correspond to each data point
            global_step (int): Global step value to record
        Shape:
            mat: :math:`(N, D)`, where N is number of data and D is feature dimension

            label_img: :math:`(N, C, H, W)`

        Examples::

            import keyword
            import torch
            meta = []
            while len(meta)<100:
                meta = meta+keyword.kwlist # get some strings
            meta = meta[:100]

            for i, v in enumerate(meta):
                meta[i] = v+str(i)

            label_img = torch.rand(100, 3, 10, 32)
            for i in range(100):
                label_img[i]*=i/100.0

            writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), metadata=meta)
        """
        if global_step == None:
            global_step = 0
            # clear pbtxt?
        save_path =  os.path.join(self.file_writer.get_logdir(), str(global_step).zfill(5))
        try:
            os.makedirs(save_path)
        except OSError:
            print('warning: Embedding dir exists, did you set global_step for add_embedding()?')
        if metadata is not None:
            assert mat.size(0) == len(metadata), '#labels should equal with #data points'
            make_tsv(metadata, save_path)
        if label_img is not None:
            assert mat.size(0) == label_img.size(0), '#images should equal with #data points'
            make_sprite(label_img, save_path)
        assert mat.dim() == 2, 'mat should be 2D, where mat.size(0) is the number of data points'
        make_mat(mat.tolist(), save_path)
        #new funcion to append to the config file a new embedding
        append_pbtxt(metadata, label_img, self.file_writer.get_logdir(), str(global_step).zfill(5))

    def close(self):
        self.file_writer.flush()
        self.file_writer.close()

    def __del__(self):
        if self.file_writer is not None:
            self.file_writer.close()

