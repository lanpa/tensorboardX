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
from .summary import scalar, histogram, image, audio, text, pr_curve
from .graph import graph
from .graph_onnx import gg
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

    def add_graph_onnx(self, graph):
        """Adds a `Graph` protocol buffer to the event file.
        """
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self._add_event(event, None)

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
            log_dir (string): save location, default is: runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each
              run. Use hierarchical folder structure to compare between runs easily. e.g. 'runs/exp1', 'runs/exp2'
            comment (string): comment that appends to the default log_dir
        """
        if not log_dir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)
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
        #
        self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
        self.scalar_dict = {}  # {writer_id : [[timestamp, step, value],...],...}

    def __append_to_scalar_dict(self, tag, scalar_value, global_step,
                                timestamp):
        """This adds an entry to the self.scalar_dict datastructure with format
        {writer_id : [[timestamp, step, value], ...], ...}.
        """
        from .x2num import makenp
        if tag not in self.scalar_dict.keys():
            self.scalar_dict[tag] = []
        self.scalar_dict[tag].append([timestamp, global_step, float(makenp(scalar_value))])

    def add_scalar(self, tag, scalar_value, global_step=None):
        """Add scalar data to summary.

        Args:
            tag (string): Data identifier
            scalar_value (float): Value to save
            global_step (int): Global step value to record
        """
        self.file_writer.add_summary(scalar(tag, scalar_value), global_step)
        self.__append_to_scalar_dict(tag, scalar_value, global_step, time.time())

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        """Adds many scalar data to summary.

        Args:
            tag (string): Data identifier
            main_tag (string): The parent name for the tags
            tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
            global_step (int): Global step value to record

        Examples::

            writer.add_scalars('run_14h',{'xsinx':i*np.sin(i/r),
                                          'xcosx':i*np.cos(i/r),
                                          'arctanx': numsteps*np.arctan(i/r)}, i)
            # This function adds three values to the same scalar plot with the tag
            # 'run_14h' in TensorBoard's scalar section.
        """
        timestamp = time.time()
        fw_logdir = self.file_writer.get_logdir()
        for tag, scalar_value in tag_scalar_dict.items():
            fw_tag = fw_logdir + "/" + main_tag + "/" + tag
            if fw_tag in self.all_writers.keys():
                fw = self.all_writers[fw_tag]
            else:
                fw = FileWriter(logdir=fw_tag)
                self.all_writers[fw_tag] = fw
            fw.add_summary(scalar(main_tag, scalar_value), global_step)
            self.__append_to_scalar_dict(fw_tag, scalar_value, global_step, timestamp)

    def export_scalars_to_json(self, path):
        """Exports to the given path an ASCII file containing all the scalars written
        so far by this instance, with the following format:
        {writer_id : [[timestamp, step, value], ...], ...}
        """
        with open(path, "w") as f:
            json.dump(self.scalar_dict, f)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow'):
        """Add histogram to summary.

        Args:
            tag (string): Data identifier
            values (numpy.array): Values to build histogram
            global_step (int): Global step value to record
            bins (string): one of {'tensorflow','auto', 'fd', ...}, this determines how the bins are made. You can find
              other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
        """
        if bins == 'tensorflow':
            bins = self.default_bins
        self.file_writer.add_summary(histogram(tag, values, bins), global_step)

    def add_image(self, tag, img_tensor, global_step=None):
        """Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (string): Data identifier
            img_tensor (torch.Tensor): Image data
            global_step (int): Global step value to record
        Shape:
            img_tensor: :math:`(3, H, W)`. Use ``torchvision.utils.make_grid()`` to prepare it is a good idea.
        """
        self.file_writer.add_summary(image(tag, img_tensor), global_step)

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100):
        """Add audio data to summary.

        Args:
            tag (string): Data identifier
            snd_tensor (torch.Tensor): Sound data
            global_step (int): Global step value to record
            sample_rate (int): sample rate in Hz

        Shape:
            snd_tensor: :math:`(1, L)`. The values should lie between [-1, 1].
        """
        self.file_writer.add_summary(audio(tag, snd_tensor, sample_rate=sample_rate), global_step)

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
            extension_dir = self.file_writer.get_logdir() + '/plugins/tensorboard_text/'
            if not os.path.exists(extension_dir):
                os.makedirs(extension_dir)
            with open(extension_dir + 'tensors.json', 'w') as fp:
                json.dump(self.text_tags, fp)

    def add_graph_onnx(self, prototxt):
        self.file_writer.add_graph_onnx(gg(prototxt))

    def add_graph(self, model, input_to_model, verbose=False):
        # prohibit second call?
        # no, let tensorboard handles it and show its warning message.
        """Add graph data to summary.

        Args:
            model (torch.nn.Module): model to draw.
            input_to_model (torch.autograd.Variable): a variable or a tuple of variables to be fed.

        """
        import torch
        from distutils.version import LooseVersion
        if LooseVersion(torch.__version__) >= LooseVersion("0.3.1"):
            pass
        else:
            if LooseVersion(torch.__version__) >= LooseVersion("0.3.0"):
                print('You are using PyTorch==0.3.0, use add_graph_onnx()')
                return
            if not hasattr(torch.autograd.Variable, 'grad_fn'):
                print('add_graph() only supports PyTorch v0.2.')
                return
        self.file_writer.add_graph(graph(model, input_to_model, verbose))

    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default'):
        """Add embedding projector data to summary.

        Args:
            mat (torch.Tensor): A matrix which each row is the feature vector of the data point
            metadata (list): A list of labels, each element will be convert to string
            label_img (torch.Tensor): Images correspond to each data point
            global_step (int): Global step value to record
            tag (string): Name for the embedding
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
        if global_step is None:
            global_step = 0
            # clear pbtxt?
        save_path = os.path.join(self.file_writer.get_logdir(), str(global_step).zfill(5))
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
        # new funcion to append to the config file a new embedding
        append_pbtxt(metadata, label_img, self.file_writer.get_logdir(), str(global_step).zfill(5), tag)

    def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None):
        """Adds precision recall curve.

        Args:
            tag (string): Data identifier
            labels (torch.Tensor): Ground truth data. Binary label for each element.
            predictions (torch.Tensor): The probability that an element be classified as true. Value should in [0, 1]
            global_step (int): Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.

        """
        from .x2num import makenp
        labels = makenp(labels)
        predictions = makenp(predictions)
        self.file_writer.add_summary(pr_curve(tag, labels, predictions, num_thresholds, weights), global_step)

    def close(self):
        if self.file_writer is None:
            return  # ignore double close
        self.file_writer.flush()
        self.file_writer.close()
        for path, writer in self.all_writers.items():
            writer.flush()
            writer.close()
        self.file_writer = self.all_writers = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
