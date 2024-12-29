
import gc
import time
from functools import wraps

from .visdom_writer import VisdomWriter
from .writer import SummaryWriter

# Supports both TensorBoard and Visdom (no embedding or graph visualization with Visdom)
vis_formats = {'tensorboard': SummaryWriter, 'visdom': VisdomWriter}


class TorchVis:
    def __init__(self, *args, **init_kwargs):
        """
        Args:
            args (list of strings): The name of the visualization target(s).
              Accepted targets are 'tensorboard' and 'visdom'.
            init_kwargs: Additional keyword parameters for the visdom writer (For example, server IP).
              See `visdom doc <https://github.com/facebookresearch/visdom/blob
              /master/README.md#visdom-arguments-python-only>`_ for more.
        """
        self.subscribers = {}
        self.register(*args, **init_kwargs)

    def register(self, *args, **init_kwargs):
        # Sets tensorboard as the default visualization format if not specified
        formats = args if args else ['tensorboard']
        for format in formats:
            if self.subscribers.get(format) is None and format in vis_formats:
                self.subscribers[format] = vis_formats[format](**init_kwargs.get(format, {}))

    def unregister(self, *args):
        for format in args:
            self.subscribers[format].close()
            del self.subscribers[format]
            gc.collect()

    def __getattr__(self, attr):
        for _, subscriber in self.subscribers.items():
            def wrapper(*args, **kwargs):
                for _, subscriber in self.subscribers.items():
                    if hasattr(subscriber, attr):
                        getattr(subscriber, attr)(*args, **kwargs)
            return wrapper
        raise AttributeError

    # Handle writer management (open/close) for the user
    def __del__(self):
        for _, subscriber in self.subscribers.items():
            subscriber.close()
