import gc

from .writer import SummaryWriter

# Supports TensorBoard visualization
vis_formats = {'tensorboard': SummaryWriter}


class TorchVis:
    def __init__(self, *args, **init_kwargs):
        """
        Args:
            args (list of strings): The name of the visualization target(s).
              Accepted targets are 'tensorboard'.
            init_kwargs: Additional keyword parameters for the writer.
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
            if format in self.subscribers:
                self.subscribers[format].close()
                del self.subscribers[format]
        gc.collect()

    def __getattr__(self, attr):
        if not self.subscribers:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

        def wrapper(*args, **kwargs):
            for _, subscriber in self.subscribers.items():
                if hasattr(subscriber, attr):
                    getattr(subscriber, attr)(*args, **kwargs)
        return wrapper

    # Handle writer management (open/close) for the user
    def __del__(self):
        for _, subscriber in self.subscribers.items():
            subscriber.close()
