"""Provides an API for reading protocol buffers from event files"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .record_reader import RecordReader
from .proto.event_pb2 import Event


class SummaryReader:
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        reader = RecordReader(self.filename)
        for event_str in reader:
            event = Event()
            event.ParseFromString(event_str)
            yield event