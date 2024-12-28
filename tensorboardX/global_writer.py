import multiprocessing as mp
from multiprocessing import Value

from .writer import SummaryWriter

global _writer
_writer = None


class GlobalSummaryWriter:
    """A class that implements an event writer that supports concurrent logging and global logging across
    different modules.

    The GlobalSummaryWriter class provides a set of API to write TensorBoard events from different processes.
    The writer instance can be accessed from different processes or modules. Also, the instance maintains
    the ``global_step`` value itself so that the interleaved requests to write an event will not conflict
    each other. This ensures that the resulting event file is TensorBoard compatible.
    With GlobalSummaryWriter, you can easily log the metrics of your parallel-trained model.
    The GlobalSummaryWriter and also be used like the ``logging`` module of Python.
    See how ``getSummaryWriter`` is used below.
    """
    def __init__(self, logdir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix='', write_to_disk=True, log_dir=None,
                 coalesce_process=True, **kwargs):
        """
        Initialize a GlobalSummaryWriter. The resulting instance will maintain a monotonically
        increasing ``global_step`` for the the event to be written. So there is no need to pass
        the global_step when calling its member functions such as ``add_scalar()``.
        All arguments for the constructor will be passed to the ordinary ``SummaryWriter.__init__()`` directly.

        Examples::

            import multiprocessing as mp
            import numpy as np
            import time
            from tensorboardX import GlobalSummaryWriter
            w = GlobalSummaryWriter()

            def train(x):
                w.add_scalar('poolmap/1', x*np.random.randn())
                time.sleep(0.05*np.random.randint(0, 10))
                w.add_scalar('poolmap/2', x*np.random.randn())

            with mp.Pool() as pool:
                pool.map(train, range(100))

        Expected result:

        .. image:: _static/img/tensorboard/add_scalar_global.png
           :scale: 50 %

        """

        self.smw = SummaryWriter(logdir=logdir, comment=comment, purge_step=purge_step, max_queue=max_queue,
                                 flush_secs=flush_secs, filename_suffix=filename_suffix, write_to_disk=write_to_disk,
                                 log_dir=log_dir)
        self.lock = mp.Lock()
        self.scalar_tag_to_step = mp.Manager().dict()
        self.image_tag_to_step = mp.Manager().dict()
        self.histogram_tag_to_step = mp.Manager().dict()
        self.text_tag_to_step = mp.Manager().dict()
        self.audio_tag_to_step = mp.Manager().dict()

    def add_scalar(self, tag, scalar_value, walltime=None):
        """Add scalar data to summary.

        Args:
            tag (string): Data identifier
            scalar_value (float): Value to save
            walltime (float): Optional override default walltime (time.time()) of event

        """
        with self.lock:
            if tag in self.scalar_tag_to_step:
                self.scalar_tag_to_step[tag] += 1
            else:
                self.scalar_tag_to_step[tag] = 0

            self.smw.add_scalar(tag, scalar_value, self.scalar_tag_to_step[tag], walltime)

    # def add_histogram(self, tag, values, bins='tensorflow', walltime=None, max_bins=None):
    #     """Add histogram to summary.

    #     Args:
    #         tag (string): Data identifier
    #         values (torch.Tensor, numpy.array): Values to build histogram
    #         bins (string): One of {'tensorflow','auto', 'fd', ...}.
    #           This determines how the bins are made. You can find
    #           other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
    #         walltime (float): Optional override default walltime (time.time()) of event

    #     """
    #     with self.new_tag_mutex.get_lock():
    #         if tag in self.histogram_tag_to_step:
    #             self.histogram_tag_to_step[tag] += 1
    #         else:
    #             self.histogram_tag_to_step[tag] = 0
    #         self.smw.add_histogram(tag,
    #                                values,
    #                                self.histogram_tag_to_step[tag],
    #                                bins=bins,
    #                                walltime=walltime,
    #                                max_bins=max_bins)

    def add_image(self, tag, img_tensor, walltime=None, dataformats='CHW'):
        """Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (string): Data identifier
            img_tensor (torch.Tensor, numpy.array): An `uint8` or `float`
                Tensor of shape `[channel, height, width]` where `channel` is 1, 3, or 4.
                The elements in img_tensor can either have values in [0, 1] (float32) or [0, 255] (uint8).
                Users are responsible to scale the data in the correct range/type.
            walltime (float): Optional override default walltime (time.time()) of event.
            dataformats (string): This parameter specifies the meaning of each dimension of the input tensor.
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or use ``add_images()`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitible as long as
            corresponding ``dataformats`` argument is passed. e.g. CHW, HWC, HW.
        """
        with self.lock:
            if tag in self.image_tag_to_step:
                self.image_tag_to_step[tag] += 1
            else:
                self.image_tag_to_step[tag] = 0

            self.smw.add_image(tag, img_tensor, self.image_tag_to_step[tag], walltime=walltime, dataformats=dataformats)

    # def add_audio(self, tag, snd_tensor, sample_rate=44100, walltime=None):
    #     """Add audio data to summary.

    #     Args:
    #         tag (string): Data identifier
    #         snd_tensor (torch.Tensor): Sound data
    #         sample_rate (int): sample rate in Hz
    #         walltime (float): Optional override default walltime (time.time()) of event
    #     Shape:
    #         snd_tensor: :math:`(1, L)`. The values should lie between [-1, 1].
    #     """

    #     with self.new_tag_mutex.get_lock():
    #         if tag in self.audio_tag_to_step:
    #             self.audio_tag_to_step[tag] += 1
    #         else:
    #             self.audio_tag_to_step[tag] = 0

    #         self.smw.add_audio(tag, snd_tensor, self.audio_tag_to_step[tag], sample_rate=44100, walltime=walltime)

    def add_text(self, tag, text_string, walltime=None):
        """Add text data to summary.

        Args:
            tag (string): Data identifier
            text_string (string): String to save
            walltime (float): Optional override default walltime (time.time()) of event

        """
        with self.lock:
            if tag in self.text_tag_to_step:
                self.text_tag_to_step[tag] += 1
            else:
                self.text_tag_to_step[tag] = 0

            self.smw.add_text(tag, text_string, global_step=self.text_tag_to_step[tag], walltime=walltime)

    @staticmethod
    def getSummaryWriter():
        """Get the writer from global namespace.

        Examples::

            # main.py
            import global_1
            import global_2

            # global1.py
            from tensorboardX import GlobalSummaryWriter
            writer = GlobalSummaryWriter.getSummaryWriter()  # This creates a new instance.
            writer.add_text('my_log', 'greeting from global1')

            # global2.py
            from tensorboardX import GlobalSummaryWriter
            writer = GlobalSummaryWriter.getSummaryWriter()  # Get the instance in global1.py.
            writer.add_text('my_log', 'greeting from global2')

        """
        global _writer
        if not hasattr(_writer, "smw") or _writer.smw is None:
            _writer = GlobalSummaryWriter()

        print("Using the global logger in:", _writer.smw.file_writer.get_logdir())
        return _writer

    @property
    def file_writer(self):
        return self.smw._get_file_writer()

    def close(self):
        self.smw.flush()
        self.smw.close()
