"""A module for visualization with tensorboard
"""

from .record_writer import RecordWriter
from .torchvis import TorchVis
from .writer import FileWriter, SummaryWriter
from .global_writer import GlobalSummaryWriter
__version__ = "2.0"  # will be overwritten if run setup.py
