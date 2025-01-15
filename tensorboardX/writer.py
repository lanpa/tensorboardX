"""Provides an API for writing protocol buffers to event files to be
consumed by TensorBoard for visualization."""


import atexit
import contextlib
import json
import logging
import os
import time
from typing import Optional, Union

import numpy

from .comet_utils import CometLogger
from .embedding import append_pbtxt, make_mat, make_sprite, make_tsv
from .event_file_writer import EventFileWriter
from .onnx_graph import load_onnx_graph
from .openvino_graph import load_openvino_graph
from .proto import event_pb2, summary_pb2
from .proto.event_pb2 import Event, SessionLog
from .summary import (
    audio,
    custom_scalars,
    histogram,
    histogram_raw,
    hparams,
    image,
    image_boxes,
    mesh,
    pr_curve,
    pr_curve_raw,
    scalar,
    text,
    video,
)
from .utils import figure_to_image

logger = logging.getLogger(__name__)

numpy_compatible = numpy.ndarray
try:
    import torch
    numpy_compatible = torch.Tensor
except ImportError:
    pass


class DummyFileWriter:
    """A fake file writer that writes nothing to the disk.
    """

    def __init__(self, logdir):
        self._logdir = logdir

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._logdir

    def add_event(self, event, step=None, walltime=None):
        return

    def add_summary(self, summary, global_step=None, walltime=None):
        return

    def add_graph(self, graph_profile, walltime=None):
        return

    def add_onnx_graph(self, graph, walltime=None):
        return

    def flush(self):
        return

    def close(self):
        return

    def reopen(self):
        return

    @contextlib.contextmanager
    def use_metadata(self, *, global_step=None, walltime=None):
        yield self


class FileWriter:
    """Writes protocol buffers to event files to be consumed by TensorBoard.

    The `FileWriter` class provides a mechanism to create an event file in a
    given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    def __init__(self, logdir, max_queue=10, flush_secs=120, filename_suffix=''):
        """Creates a `FileWriter` and an event file.
        On construction the writer creates a new event file in `logdir`.
        The other arguments to the constructor control the asynchronous writes to
        the event file.

        Args:
          logdir: A string. Directory where event file will be written.
          max_queue: Integer. Size of the queue for pending events and
            summaries before one of the 'add' calls forces a flush to disk.
            Default is ten items.
          flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk. Default is every two minutes.
          filename_suffix: A string. Suffix added to all event filenames
            in the logdir directory. More details on filename construction in
            tensorboard.summary.writer.event_file_writer.EventFileWriter.
        """
        # Sometimes PosixPath is passed in and we need to coerce it to
        # a string in all cases
        # TODO: See if we can remove this in the future if we are
        # actually the ones passing in a PosixPath
        logdir = str(logdir)
        self.event_writer = EventFileWriter(
            logdir, max_queue, flush_secs, filename_suffix)

        def cleanup():
            self.event_writer.close()

        atexit.register(cleanup)
        self._default_metadata = {}

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self.event_writer.get_logdir()

    def add_event(self, event, step=None, walltime=None):
        """Adds an event to the event file.
        Args:
          event: An `Event` protocol buffer.
          step: Number. Optional global step value for training process
            to record with the event.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time())
        """
        walltime = (
            self._default_metadata.get("walltime", time.time())
            if walltime is None
            else walltime
        )
        if walltime is not None:
            event.wall_time = walltime
        step = self._default_metadata.get("global_step") if step is None else step
        if step is not None:
            # Make sure step is converted from numpy or other formats
            # since protobuf might not convert depending on version
            event.step = int(step)
        self.event_writer.add_event(event)

    def add_summary(self, summary, global_step=None, walltime=None):
        """Adds a `Summary` protocol buffer to the event file.
        This method wraps the provided summary in an `Event` protocol buffer
        and adds it to the event file.

        Args:
          summary: A `Summary` protocol buffer.
          global_step: Number. Optional global step value for training process
            to record with the summary.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time())
        """
        event = event_pb2.Event(summary=summary)
        self.add_event(event, global_step, walltime)

    def add_graph(self, graph_profile, walltime=None):
        """Adds a `Graph` and step stats protocol buffer to the event file.

        Args:
          graph_profile: A `Graph` and step stats protocol buffer.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        graph = graph_profile[0]
        stepstats = graph_profile[1]
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)

        trm = event_pb2.TaggedRunMetadata(
            tag='profiler', run_metadata=stepstats.SerializeToString())
        event = event_pb2.Event(tagged_run_metadata=trm)
        self.add_event(event, None, walltime)

    def add_onnx_graph(self, graph, walltime=None):
        """Adds a `Graph` protocol buffer to the event file.

        Args:
          graph: A `Graph` protocol buffer.
          walltime: float. Optional walltime to override the default (current)
            _get_file_writerfrom time.time())
        """
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)

    def add_openvino_graph(self, graph, walltime=None):
        """Adds a `Graph` protocol buffer to the event file.

        Args:
          graph: A `Graph` protocol buffer.
          walltime: float. Optional walltime to override the default (current)
            _get_file_writerfrom time.time())
        """
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)

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

    @contextlib.contextmanager
    def use_metadata(self, *, global_step=None, walltime=None):
        """Context manager to temporarily set default metadata for all enclosed :meth:`add_event`
        calls.

        Args:
            global_step: Global step value to record
            walltime: Walltime to record (defaults to time.time())

        Examples::

            with writer.use_metadata(global_step=10):
                writer.add_event(event)
        """
        assert not self._default_metadata, "Default metadata is already set."
        assert (
            global_step is not None or walltime is not None
        ), "At least one of `global_step` or `walltime` must be provided."
        self._default_metadata = {"global_step": global_step, "walltime": walltime}
        yield self
        self._default_metadata = {}


class SummaryWriter:
    """Writes entries directly to event files in the logdir to be
    consumed by TensorBoard.

    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    def __init__(
            self,
            logdir: Optional[str] = None,
            comment: Optional[str] = "",
            purge_step: Optional[int] = None,
            max_queue: Optional[int] = 10,
            flush_secs: Optional[int] = 120,
            filename_suffix: Optional[str] = '',
            write_to_disk: Optional[bool] = True,
            log_dir: Optional[str] = None,
            comet_config: Optional[dict] = {"disabled": True},
            **kwargs):
        """Creates a `SummaryWriter` that will write out events and summaries
        to the event file.

        Args:
            logdir: Save directory location. Default is
              runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run.
              Use hierarchical folder structure to compare
              between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
              for each new experiment to compare across them.
            comment: Comment logdir suffix appended to the default
              ``logdir``. If ``logdir`` is assigned, this argument has no effect.
            purge_step:
              When logging crashes at step :math:`T+X` and restarts at step :math:`T`,
              any events whose global_step larger or equal to :math:`T` will be
              purged and hidden from TensorBoard.
              Note that crashed and resumed experiments should have the same ``logdir``.
            max_queue: Size of the queue for pending events and
              summaries before one of the 'add' calls forces a flush to disk.
              Default is ten items.
            flush_secs: How often, in seconds, to flush the
              pending events and summaries to disk. Default is every two minutes.
            filename_suffix: Suffix added to all event filenames in
              the logdir directory. More details on filename construction in
              tensorboard.summary.writer.event_file_writer.EventFileWriter.
            write_to_disk:
              If pass `False`, SummaryWriter will not write to disk.
            comet_config:
              A comet config dictionary. Contains parameters that need to be
              passed to comet like workspace, project_name, api_key, disabled etc

        Examples::

            from tensorboardX import SummaryWriter

            # create a summary writer with automatically generated folder name.
            writer = SummaryWriter()
            # folder location: runs/May04_22-14-54_s-MacBook-Pro.local/

            # create a summary writer using the specified folder name.
            writer = SummaryWriter("my_experiment")
            # folder location: my_experiment

            # create a summary writer with comment appended.
            writer = SummaryWriter(comment="LR_0.1_BATCH_16")
            # folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/

        """
        if log_dir is not None and logdir is None:
            logdir = log_dir
        if not logdir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            logdir = os.path.join(
                'runs', current_time + '_' + socket.gethostname() + comment)
        self.logdir = logdir
        self.purge_step = purge_step
        self._max_queue = max_queue
        self._flush_secs = flush_secs
        self._filename_suffix = filename_suffix
        self._write_to_disk = write_to_disk
        self._comet_config = comet_config
        self._comet_logger = None
        self.kwargs = kwargs

        # Initialize the file writers, but they can be cleared out on close
        # and recreated later as needed.
        self.file_writer = self.all_writers = None
        self._get_file_writer()

        # Create default bins for histograms, see generate_testdata.py in tensorflow/tensorboard
        v = 1E-12
        buckets = []
        neg_buckets = []
        while v < 1E20:
            buckets.append(v)
            neg_buckets.append(-v)
            v *= 1.1
        self.default_bins = neg_buckets[::-1] + [0] + buckets

        self.scalar_dict = {}
        self._default_metadata = {}

    def __append_to_scalar_dict(self, tag, scalar_value, global_step,
                                timestamp):
        """This adds an entry to the self.scalar_dict datastructure with format
        {writer_id : [[timestamp, step, value], ...], ...}.
        """
        from .x2num import make_np
        if tag not in self.scalar_dict:
            self.scalar_dict[tag] = []
        self.scalar_dict[tag].append(
            [timestamp, global_step, float(make_np(scalar_value).squeeze())])

    def _get_file_writer(self):
        """Returns the default FileWriter instance. Recreates it if closed."""
        if not self._write_to_disk:
            self.file_writer = DummyFileWriter(logdir=self.logdir)
            self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
            return self.file_writer

        if self.all_writers is None or self.file_writer is None:
            self.file_writer = FileWriter(logdir=self.logdir,
                                          max_queue=self._max_queue,
                                          flush_secs=self._flush_secs,
                                          filename_suffix=self._filename_suffix,
                                          **self.kwargs)
            if self.purge_step is not None:
                self.file_writer.add_event(
                    Event(step=self.purge_step, file_version='brain.Event:2'))
                self.file_writer.add_event(
                    Event(step=self.purge_step, session_log=SessionLog(status=SessionLog.START)))
            self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
        return self.file_writer

    def _get_comet_logger(self):
        """Returns a comet logger instance. Recreates it if closed."""
        if self._comet_logger is None:
            self._comet_logger = CometLogger(self._comet_config)
        return self._comet_logger

    def add_hparams(
            self,
            hparam_dict: dict[str, Union[bool, str, float, int]],
            metric_dict: dict[str, float],
            name: Optional[str] = None,
            global_step: Optional[int] = None):
        """Add a set of hyperparameters to be compared in tensorboard.

        Args:
            hparam_dict: Each key-value pair in the dictionary is the
              name of the hyper parameter and it's corresponding value.
            metric_dict: Each key-value pair in the dictionary is the
              name of the metric and it's corresponding value.
              Note that the key used here should be unique in the
              tensorboard record. Otherwise the value you added by `add_scalar`
              will be displayed in hparam plugin. In most
              cases, this is unwanted.
            name: Personnalised name of the hparam session
            global_step: Current time step

        Examples::

            from tensorboardX import SummaryWriter
            with SummaryWriter() as w:
                for i in range(5):
                    w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})

        Expected result:

        .. image:: _static/img/tensorboard/add_hparam.png
           :scale: 50 %
        """
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        if not name:
            name = str(time.time())

        with SummaryWriter(logdir=os.path.join(self.file_writer.get_logdir(), name)) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v, global_step)
        self._get_comet_logger().log_parameters(hparam_dict, step=global_step)

    def add_scalar(
            self,
            tag: str,
            scalar_value: Union[float, numpy_compatible],
            global_step: Optional[int] = None,
            walltime: Optional[float] = None,
            display_name: Optional[str] = "",
            summary_description: Optional[str] = ""):
        """Add scalar data to summary.

        Args:
            tag: Data identifier
            scalar_value: Value to be logged in tensorboard
            global_step: Global step value to record
            walltime: Optional override default walltime (time.time()) of event
            display_name: The title of the plot. If empty string is passed,
              `tag` will be used.
            summary_description: The comprehensive text that will showed
              by clicking the information icon on TensorBoard.
        Examples::

            from tensorboardX import SummaryWriter
            writer = SummaryWriter()
            x = range(100)
            for i in x:
                writer.add_scalar('y=2x', i * 2, i)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_scalar.png
           :scale: 50 %

        """
        self._get_file_writer().add_summary(
            scalar(tag, scalar_value, display_name, summary_description), global_step, walltime)
        self._get_comet_logger().log_metric(tag, display_name, scalar_value, global_step)

    def add_scalars(
            self,
            main_tag: str,
            tag_scalar_dict: dict[str, float],
            global_step: Optional[int] = None,
            walltime: Optional[float] = None):
        """Adds many scalar data to summary.

        Note that this function also keeps logged scalars in memory. In extreme case it explodes your RAM.

        Args:
            main_tag: The parent name for the tags
            tag_scalar_dict: Key-value pair storing the tag and corresponding values
            global_step: Global step value to record
            walltime: Optional override default walltime (time.time()) of event

        Examples::

            from tensorboardX import SummaryWriter
            writer = SummaryWriter()
            r = 5
            for i in range(100):
                writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                                'xcosx':i*np.cos(i/r),
                                                'tanx': np.tan(i/r)}, i)
            writer.close()
            # This call adds three values to the same scalar plot with the tag
            # 'run_14h' in TensorBoard's scalar section.

        Expected result:

        .. image:: _static/img/tensorboard/add_scalars.png
           :scale: 50 %

        """
        walltime = time.time() if walltime is None else walltime
        fw_logdir = self._get_file_writer().get_logdir()
        for tag, scalar_value in tag_scalar_dict.items():
            fw_tag = os.path.join(str(fw_logdir), main_tag, tag)
            if fw_tag in self.all_writers:
                fw = self.all_writers[fw_tag]
            else:
                fw = FileWriter(logdir=fw_tag)
                self.all_writers[fw_tag] = fw
            fw.add_summary(scalar(main_tag, scalar_value),
                           global_step, walltime)
            self.__append_to_scalar_dict(
                fw_tag, scalar_value, global_step, walltime)
        self._get_comet_logger().log_metrics(tag_scalar_dict, main_tag, step=global_step)

    def export_scalars_to_json(self, path):
        """Exports to the given path an ASCII file containing all the scalars written
        so far by this instance, with the following format:
        {writer_id : [[timestamp, step, value], ...], ...}

        The scalars saved by ``add_scalars()`` will be flushed after export.
        """
        with open(path, "w") as f:
            json.dump(self.scalar_dict, f)
        self.scalar_dict = {}

    def add_histogram(
            self,
            tag: str,
            values: numpy_compatible,
            global_step: Optional[int] = None,
            bins: Optional[str] = 'tensorflow',
            walltime: Optional[float] = None,
            max_bins=None):
        """Add histogram to summary.

        Args:
            tag: Data identifier
            values: Values to build histogram
            global_step: Global step value to record
            bins: One of {'tensorflow','auto', 'fd', ...}. This determines how the
              bins are made. You can find other options in the `numpy reference
              <https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html>`_.
            walltime: Optional override default walltime (time.time()) of event

        Examples::

            from tensorboardX import SummaryWriter
            import numpy as np
            writer = SummaryWriter()
            for i in range(10):
                x = np.random.random(1000)
                writer.add_histogram('distribution centers', x + i, i)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_histogram.png
           :scale: 50 %

        """
        if isinstance(bins, str) and bins == 'tensorflow':
            bins = self.default_bins
        self._get_file_writer().add_summary(
            histogram(tag, values, bins, max_bins=max_bins), global_step, walltime)
        self._get_comet_logger().log_histogram(values, tag, global_step)

    def add_histogram_raw(
            self,
            tag: str,
            min,
            max,
            num,
            sum,
            sum_squares,
            bucket_limits,
            bucket_counts,
            global_step: Optional[int] = None,
            walltime: Optional[float] = None):
        """Adds histogram with raw data.

        Args:
            tag: Data identifier
            min (float or int): Min value
            max (float or int): Max value
            num (int): Number of values
            sum (float or int): Sum of all values
            sum_squares (float or int): Sum of squares for all values
            bucket_limits (torch.Tensor, numpy.array): Upper value per
              bucket, note that the bucket_limits returned from `np.histogram`
              has one more element. See the comment in the following example.
            bucket_counts (torch.Tensor, numpy.array): Number of values per bucket
            global_step: Global step value to record
            walltime: Optional override default walltime (time.time()) of event

        Examples::

            import numpy as np
            dummy_data = []
            for idx, value in enumerate(range(30)):
                dummy_data += [idx + 0.001] * value
            values = np.array(dummy_data).astype(float).reshape(-1)
            counts, limits = np.histogram(values)
            sum_sq = values.dot(values)
            with SummaryWriter() as summary_writer:
                summary_writer.add_histogram_raw(
                        tag='hist_dummy_data',
                        min=values.min(),
                        max=values.max(),
                        num=len(values),
                        sum=values.sum(),
                        sum_squares=sum_sq,
                        bucket_limits=limits[1:].tolist(),  # <- note here.
                        bucket_counts=counts.tolist(),
                        global_step=0)

        """
        if len(bucket_limits) != len(bucket_counts):
            raise ValueError('len(bucket_limits) != len(bucket_counts), see the document.')
        summary = histogram_raw(tag,
                                min,
                                max,
                                num,
                                sum,
                                sum_squares,
                                bucket_limits,
                                bucket_counts)
        self._get_file_writer().add_summary(
            summary,
            global_step,
            walltime)
        self._get_comet_logger().log_histogram_raw(tag, summary, step=global_step)

    def add_image(
            self,
            tag: str,
            img_tensor: numpy_compatible,
            global_step: Optional[int] = None,
            walltime: Optional[float] = None,
            dataformats: Optional[str] = 'CHW'):
        """Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag: Data identifier
            img_tensor: An `uint8` or `float` Tensor of shape `
                [channel, height, width]` where `channel` is 1, 3, or 4.
                The elements in img_tensor can either have values
                in [0, 1] (float32) or [0, 255] (uint8).
                Users are responsible to scale the data in the correct range/type.
            global_step: Global step value to record
            walltime: Optional override default walltime (time.time()) of event.
            dataformats: This parameter specifies the meaning of each dimension of the input tensor.
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or use ``add_images()`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitible as long as
            corresponding ``dataformats`` argument is passed. e.g. CHW, HWC, HW.

        Examples::

            from tensorboardX import SummaryWriter
            import numpy as np
            img = np.zeros((3, 100, 100))
            img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
            img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

            img_HWC = np.zeros((100, 100, 3))
            img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
            img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

            writer = SummaryWriter()
            writer.add_image('my_image', img, 0)

            # If you have non-default dimension setting, set the dataformats argument.
            writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_image.png
           :scale: 50 %

        """
        summary = image(tag, img_tensor, dataformats=dataformats)
        encoded_image_string = summary.value[0].image.encoded_image_string
        self._get_file_writer().add_summary(
            summary, global_step, walltime)
        self._get_comet_logger().log_image_encoded(encoded_image_string, tag, step=global_step)

    def add_images(
            self,
            tag: str,
            img_tensor: numpy_compatible,
            global_step: Optional[int] = None,
            walltime: Optional[float] = None,
            dataformats: Optional[str] = 'NCHW'):
        """Add batched (4D) image data to summary.
        Besides passing 4D (NCHW) tensor, you can also pass a list of tensors of the same size.
        In this case, the ``dataformats`` should be `CHW` or `HWC`.
        Note that this requires the ``pillow`` package.

        Args:
            tag: Data identifier
            img_tensor: Image data
                The elements in img_tensor can either have values in [0, 1] (float32) or [0, 255] (uint8).
                Users are responsible to scale the data in the correct range/type.
            global_step: Global step value to record
            walltime: Optional override default walltime (time.time()) of event
        Shape:
            img_tensor: Default is :math:`(N, 3, H, W)`. If ``dataformats`` is specified, other shape will be
            accepted. e.g. NCHW or NHWC.

        Examples::

            from tensorboardX import SummaryWriter
            import numpy as np

            img_batch = np.zeros((16, 3, 100, 100))
            for i in range(16):
                img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
                img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

            writer = SummaryWriter()
            writer.add_images('my_image_batch', img_batch, 0)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_images.png
           :scale: 30 %

        """
        if isinstance(img_tensor, list):  # a list of tensors in CHW or HWC
            if dataformats.upper() != 'CHW' and dataformats.upper() != 'HWC':
                print('A list of image is passed, but the dataformat is neither CHW nor HWC.')
                print('Nothing is written.')
                return
            import torch
            try:
                img_tensor = torch.stack(img_tensor, 0)
            except TypeError:
                import numpy as np
                img_tensor = np.stack(img_tensor, 0)

            dataformats = 'N' + dataformats

        summary = image(tag, img_tensor, dataformats=dataformats)
        encoded_image_string = summary.value[0].image.encoded_image_string
        self._get_file_writer().add_summary(
            summary, global_step, walltime)
        self._get_comet_logger().log_image_encoded(encoded_image_string, tag, step=global_step)

    def add_image_with_boxes(
            self,
            tag: str,
            img_tensor: numpy_compatible,
            box_tensor: numpy_compatible,
            global_step: Optional[int] = None,
            walltime: Optional[float] = None,
            dataformats: Optional[str] = 'CHW',
            labels: Optional[list[str]] = None,
            **kwargs):
        """Add image and draw bounding boxes on the image.

        Args:
            tag: Data identifier
            img_tensor: Image data
            box_tensor: Box data (for detected objects)
              box should be represented as [x1, y1, x2, y2].
            global_step: Global step value to record
            walltime: override default walltime (time.time()) of event
            labels: The strings to be show on each bounding box.
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. It can be specified with ``dataformats`` argument.
            e.g. CHW or HWC

            box_tensor: (torch.Tensor, numpy.array, or string/blobname): NX4,  where N is the number of
            boxes and each 4 elements in a row represents (xmin, ymin, xmax, ymax).
        """
        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]
            if len(labels) != box_tensor.shape[0]:
                logger.warning('Number of labels do not equal to number of box, skip the labels.')
                labels = None
        summary = image_boxes(
            tag, img_tensor, box_tensor, dataformats=dataformats, labels=labels, **kwargs)
        encoded_image_string = summary.value[0].image.encoded_image_string
        self._get_file_writer().add_summary(
            summary, global_step, walltime)
        self._get_comet_logger().log_image_encoded(encoded_image_string, tag, step=global_step)

    def add_figure(
            self,
            tag: str,
            figure,
            global_step: Optional[int] = None,
            close: Optional[bool] = True,
            walltime: Optional[float] = None):
        """Render matplotlib figure into an image and add it to summary.

        Note that this requires the ``matplotlib`` package.

        Args:
            tag: Data identifier
            figure (matplotlib.pyplot.figure) or list of figures: Figure or a list of figures
            global_step: Global step value to record
            close: Flag to automatically close the figure
            walltime: Override default walltime (time.time()) of event
        """
        if isinstance(figure, list):
            self.add_image(tag, figure_to_image(figure, close), global_step, walltime, dataformats='NCHW')
        else:
            self.add_image(tag, figure_to_image(figure, close), global_step, walltime, dataformats='CHW')

    def add_video(
            self,
            tag: str,
            vid_tensor: numpy_compatible,
            global_step: Optional[int] = None,
            fps: Optional[Union[int, float]] = 4,
            walltime: Optional[float] = None,
            dataformats: Optional[str] = 'NTCHW'):
        """Add video data to summary.

        Note that this requires the ``moviepy`` package.

        Args:
            tag: Data identifier
            vid_tensor: Video data
            global_step: Global step value to record
            fps: Frames per second
            walltime: Optional override default walltime (time.time()) of event
            dataformats: Specify different permutation of the video tensor
        Shape:
            vid_tensor: :math:`(N, T, C, H, W)`. The values should lie in [0, 255]
            for type `uint8` or [0, 1] for type `float`.
        """
        summary = video(tag, vid_tensor, fps, dataformats=dataformats)
        encoded_image_string = summary.value[0].image.encoded_image_string
        self._get_file_writer().add_summary(
            summary, global_step, walltime)
        self._get_comet_logger().log_image_encoded(encoded_image_string, tag, step=global_step)

    def add_audio(
            self,
            tag: str,
            snd_tensor: numpy_compatible,
            global_step: Optional[int] = None,
            sample_rate: Optional[int] = 44100,
            walltime: Optional[float] = None):
        """Add audio data to summary.

        Args:
            tag: Data identifier
            snd_tensor: Sound data
            global_step: Global step value to record
            sample_rate: sample rate in Hz
            walltime: Optional override default walltime (time.time()) of event
        Shape:
            snd_tensor: :math:`(L, C)`. The values should lie between [-1, 1].
            Where `L` is the number of audio frames and `C` is the channel. Set
            channel equals to 2 for stereo.
        """
        self._get_file_writer().add_summary(
            audio(tag, snd_tensor, sample_rate=sample_rate), global_step, walltime)
        self._get_comet_logger().log_audio(snd_tensor, sample_rate, tag, step=global_step)

    def add_text(
            self,
            tag: str,
            text_string: str,
            global_step: Optional[int] = None,
            walltime: Optional[float] = None):
        """Add text data to summary.

        Args:
            tag: Data identifier
            text_string: String to save
            global_step: Global step value to record
            walltime: Optional override default walltime (time.time()) of event
        Examples::

            writer.add_text('lstm', 'This is an lstm', 0)
            writer.add_text('rnn', 'This is an rnn', 10)
        """
        self._get_file_writer().add_summary(
            text(tag, text_string), global_step, walltime)
        self._get_comet_logger().log_text(text_string, global_step)

    def add_onnx_graph(
            self,
            onnx_model_file):
        """Add onnx graph to TensorBoard.

        Args:
            onnx_model_file (string): The path to the onnx model.
        """
        self._get_file_writer().add_onnx_graph(load_onnx_graph(onnx_model_file))
        self._get_comet_logger().log_asset(onnx_model_file)

    def add_openvino_graph(
            self,
            xmlname):
        """Add openvino graph to TensorBoard.

        Args:
            xmlname (string): The path to the openvino model. (the xml file)
        """
        self._get_file_writer().add_openvino_graph(load_openvino_graph(xmlname))
        self._get_comet_logger().log_asset(xmlname)

    def add_graph(
            self,
            model,
            input_to_model=None,
            verbose=False,
            use_strict_trace=True):
        """Add graph data to summary. The graph is actually processed by `torch.utils.tensorboard.add_graph()`

        Args:
            model (torch.nn.Module): Model to draw.
            input_to_model (torch.Tensor or list of torch.Tensor): A variable or a tuple of
                variables to be fed.
            verbose (bool): Whether to print graph structure in console.
            use_strict_trace (bool): Whether to pass keyword argument `strict` to
                `torch.jit.trace`. Pass False when you want the tracer to
                record your mutable container types (list, dict)
        """
        from torch.utils.tensorboard._pytorch_graph import graph
        self._get_file_writer().add_graph(graph(model, input_to_model, verbose, use_strict_trace=use_strict_trace))

    @staticmethod
    def _encode(rawstr):
        # I'd use urllib but, I'm unsure about the differences from python3 to python2, etc.
        retval = rawstr
        retval = retval.replace("%", "%{:02x}".format(ord("%")))
        retval = retval.replace("/", "%{:02x}".format(ord("/")))
        retval = retval.replace("\\", "%{:02x}".format(ord("\\")))
        return retval

    def add_embedding(
            self,
            mat: numpy_compatible,
            metadata=None,
            label_img: numpy_compatible = None,
            global_step: Optional[int] = None,
            tag='default',
            metadata_header=None):
        r"""Add embedding projector data to summary.

        Args:
            mat: A matrix which each row is the feature vector of the data point
            metadata (list): A list of labels, each element will be converted to
                string.
            label_img: Images correspond to each
                data point. Each image should be square sized. The amount and
                the size of the images are limited by the Tensorboard frontend,
                see limits below.
            global_step: Global step value to record
            tag: Name for the embedding
        Shape:
            mat: :math:`(N, D)`, where N is number of data and D is feature dimension

            label_img: :math:`(N, C, H, W)`, where `Height` should be equal to `Width`.
            Also, :math:`\sqrt{N}*W` must be less than or equal to 8192, so that the generated sprite
            image can be loaded by the Tensorboard frontend
            (see `tensorboardX#516 <https://github.com/lanpa/tensorboardX/issues/516>`_ for more).

        Examples::

            import keyword
            import torch
            meta = []
            while len(meta)<100:
                meta = meta+keyword.kwlist # get some strings
            meta = meta[:100]

            for i, v in enumerate(meta):
                meta[i] = v+str(i)

            label_img = torch.rand(100, 3, 32, 32)
            for i in range(100):
                label_img[i]*=i/100.0

            writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), metadata=meta)
        """

        # programmer's note: This function has nothing to do with event files.
        # The hard-coded projector_config.pbtxt is the only source for TensorBoard's
        # current implementation. (as of Dec. 2019)
        from .x2num import make_np
        mat = make_np(mat)
        if global_step is None:
            global_step = 0
            # clear pbtxt?
        # Maybe we should encode the tag so slashes don't trip us up?
        # I don't think this will mess us up, but better safe than sorry.
        subdir = f"{str(global_step).zfill(5)}/{self._encode(tag)}"
        save_path = os.path.join(self._get_file_writer().get_logdir(), subdir)
        try:
            os.makedirs(save_path)
        except OSError:
            print(
                'warning: Embedding dir exists, did you set global_step for add_embedding()?')
        if metadata is not None:
            assert mat.shape[0] == len(
                metadata), '#labels should equal with #data points'
            make_tsv(metadata, save_path, metadata_header=metadata_header)
        if label_img is not None:
            assert mat.shape[0] == label_img.shape[0], '#images should equal with #data points'
            assert label_img.shape[2] == label_img.shape[3], 'Image should be square, see tensorflow/tensorboard#670'
            make_sprite(label_img, save_path)
        assert mat.ndim == 2, 'mat should be 2D, where mat.size(0) is the number of data points'
        make_mat(mat, save_path)
        # new funcion to append to the config file a new embedding
        append_pbtxt(metadata, label_img,
                     self._get_file_writer().get_logdir(), subdir, global_step, tag)
        template_filename = f'{tag}.json' if tag is not None else None

        self._get_comet_logger().log_embedding(mat, metadata, label_img, template_filename=template_filename)

    def add_pr_curve(
            self,
            tag: str,
            labels: numpy_compatible,
            predictions: numpy_compatible,
            global_step: Optional[int] = None,
            num_thresholds: Optional[int] = 127,
            weights=None,
            walltime: Optional[float] = None):
        """Adds precision recall curve.
        Plotting a precision-recall curve lets you understand your model's
        performance under different threshold settings. With this function,
        you provide the ground truth labeling (T/F) and prediction confidence
        (usually the output of your model) for each target. The TensorBoard UI
        will let you choose the threshold interactively.

        Args:
            tag: Data identifier
            labels: Ground truth data. Binary label for each element.
            predictions:
              The probability that an element be classified as true.
              Value should in [0, 1]
            global_step: Global step value to record
            num_thresholds: Number of thresholds used to draw the curve.
            walltime: Override default walltime (time.time()) of event

        Examples::

            from tensorboardX import SummaryWriter
            import numpy as np
            labels = np.random.randint(2, size=100)  # binary label
            predictions = np.random.rand(100)
            writer = SummaryWriter()
            writer.add_pr_curve('pr_curve', labels, predictions, 0)
            writer.close()

        """
        from .x2num import make_np
        labels, predictions = make_np(labels), make_np(predictions)

        summary = pr_curve(tag, labels, predictions, num_thresholds, weights)
        self._get_file_writer().add_summary(
            summary,
            global_step, walltime)

        self._get_comet_logger().log_pr_data(tag, summary, num_thresholds, step=global_step)

    def add_pr_curve_raw(
            self,
            tag: str,
            true_positive_counts: numpy_compatible,
            false_positive_counts: numpy_compatible,
            true_negative_counts: numpy_compatible,
            false_negative_counts: numpy_compatible,
            precision: numpy_compatible,
            recall: numpy_compatible,
            global_step: Optional[int] = None,
            num_thresholds: Optional[int] = 127,
            weights=None,
            walltime: Optional[float] = None):
        """Adds precision recall curve with raw data.

        Args:
            tag: Data identifier
            global_step: Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.
            walltime: Optional override default walltime (time.time()) of event
              see: `Tensorboard refenence
              <https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md>`_
        """
        self._get_file_writer().add_summary(
            pr_curve_raw(tag,
                         true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         num_thresholds,
                         weights),
            global_step,
            walltime)
        self._get_comet_logger().log_pr_raw_data(tag, step=global_step,
                                                 true_positive_counts=true_positive_counts,
                                                 false_positive_counts=false_positive_counts,
                                                 true_negative_counts=true_negative_counts,
                                                 false_negative_counts=false_negative_counts,
                                                 precision=precision,
                                                 recall=recall,
                                                 num_thresholds=num_thresholds,
                                                 weights=weights)

    def add_custom_scalars_multilinechart(
            self,
            tags: list[str],
            category: str = 'default',
            title: str = 'untitled'):
        """Shorthand for creating multilinechart. Similar to ``add_custom_scalars()``, but the only necessary argument
        is *tags*.

        Args:
            tags: list of tags that have been used in ``add_scalar()``

        Examples::

            writer.add_custom_scalars_multilinechart(['twse/0050', 'twse/2330'])
        """
        layout = {category: {title: ['Multiline', tags]}}
        self._get_file_writer().add_summary(custom_scalars(layout))

    def add_custom_scalars_marginchart(
            self,
            tags: list[str],
            category: str = 'default',
            title: str = 'untitled'):
        """Shorthand for creating marginchart. Similar to ``add_custom_scalars()``, but the only necessary argument
        is *tags*, which should have exactly 3 elements.

        Args:
            tags: list of tags that have been used in ``add_scalar()``

        Examples::

            writer.add_custom_scalars_marginchart(['twse/0050', 'twse/2330', 'twse/2006'])
        """
        assert len(tags) == 3
        layout = {category: {title: ['Margin', tags]}}
        self._get_file_writer().add_summary(custom_scalars(layout))

    def add_custom_scalars(
            self,
            layout: dict[str, dict[str, list]]):
        """Create special chart by collecting charts tags in 'scalars'. Note that this function can only be called once
        for each SummaryWriter() object. Because it only provides metadata to tensorboard, the function can be called
        before or after the training loop. See ``examples/demo_custom_scalars.py`` for more.

        Args:
            layout: {categoryName: *charts*}, where *charts* is also a dictionary
              {chartName: *ListOfProperties*}. The first element in *ListOfProperties* is the chart's type
              (one of **Multiline** or **Margin**) and the second element should be a list containing the tags
              you have used in add_scalar function, which will be collected into the new chart.

        Examples::

            layout = {'Taiwan':{'twse':['Multiline',['twse/0050', 'twse/2330']]},
                         'USA':{ 'dow':['Margin',   ['dow/aaa', 'dow/bbb', 'dow/ccc']],
                              'nasdaq':['Margin',   ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]}}

            writer.add_custom_scalars(layout)
        """
        self._get_file_writer().add_summary(custom_scalars(layout))

    def add_mesh(
            self,
            tag: str,
            vertices: numpy_compatible,
            colors: numpy_compatible = None,
            faces: numpy_compatible = None,
            config_dict=None,
            global_step: Optional[int] = None,
            walltime: Optional[float] = None):
        """Add meshes or 3D point clouds to TensorBoard. The visualization is based on Three.js,
        so it allows users to interact with the rendered object. Besides the basic definitions
        such as vertices, faces, users can further provide camera parameter, lighting condition, etc.
        Please check https://threejs.org/docs/index.html#manual/en/introduction/Creating-a-scene for
        advanced usage.

        Args:
            tag: Data identifier
            vertices: List of the 3D coordinates of vertices.
            colors: Colors for each vertex
            faces: Indices of vertices within each triangle. (Optional)
            config_dict: Dictionary with ThreeJS classes names and configuration.
            global_step: Global step value to record
            walltime: Optional override default walltime (time.time())
              seconds after epoch of event

        Shape:
            vertices: :math:`(B, N, 3)`. (batch, number_of_vertices, channels). If
              Nothing show on tensorboard, try normalizing the values to [-1, 1].

            colors: :math:`(B, N, 3)`. The values should lie in [0, 255].

            faces: :math:`(B, N, 3)`. The values should lie in [0, number_of_vertices] for type `uint8`.

        Expected result after running ``examples/demo_mesh.py``:

        .. image:: _static/img/tensorboard/add_mesh.png
           :scale: 30 %

        """
        self._get_file_writer().add_summary(mesh(tag, vertices, colors, faces, config_dict), global_step, walltime)
        self._get_comet_logger().log_mesh(tag, vertices, colors, faces,
                                          config_dict, global_step, walltime)

    def close(self):
        """Close the current SummaryWriter. This call flushes the unfinished write operation.
        Use context manager (with statement) whenever it's possible.
        """
        if self.all_writers is None:
            return  # ignore double close
        for writer in self.all_writers.values():
            writer.flush()
            writer.close()
        self.file_writer = self.all_writers = None
        self._get_comet_logger().end()
        self._comet_logger = None

    def flush(self):
        """Force the data in memory to be flushed to disk. Use this call if tensorboard does not update reqularly.
        Another way is to set the `flush_secs` when creating the SummaryWriter.
        """
        if self.all_writers is None:
            return  # ignore double close
        for writer in self.all_writers.values():
            writer.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @contextlib.contextmanager
    def use_metadata(self, *, global_step=None, walltime=None):
        """Context manager to temporarily set default metadata for all enclosed :meth:`add_*` calls.

        Args:
            global_step: Global step value to record
            walltime: Walltime to record (defaults to time.time())

        Examples::

            with writer.use_metadata(global_step=10):
                writer.add_scalar("loss", 3.0)
        """
        with self._get_file_writer().use_metadata(global_step=global_step, walltime=walltime):
            yield self
