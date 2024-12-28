"""A module for visualization with tensorboard
"""

from .global_writer import GlobalSummaryWriter
from .record_writer import RecordWriter
from .torchvis import TorchVis
from .writer import FileWriter, SummaryWriter

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown version"
