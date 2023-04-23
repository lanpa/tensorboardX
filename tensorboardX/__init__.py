"""A module for visualization with tensorboard
"""

from .record_writer import RecordWriter
from .torchvis import TorchVis
from .writer import FileWriter, SummaryWriter
from .global_writer import GlobalSummaryWriter

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown version"
