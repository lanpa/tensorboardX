import os
import struct
import tempfile
import threading
import unittest
from unittest import mock

from tensorboardX.event_file_writer import EventsWriter
from tensorboardX.proto import event_pb2
from tensorboardX.proto.summary_pb2 import Summary


def _read_tfrecord_events(path):
    events = []
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if not header:
                break
            length = struct.unpack("Q", header)[0]
            f.read(4)  # length crc
            payload = f.read(length)
            f.read(4)  # payload crc
            ev = event_pb2.Event()
            ev.ParseFromString(payload)
            events.append(ev)
    return events


class EventFileWriterNamesTest(unittest.TestCase):
    def test_same_second_events_writers_use_distinct_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "events")
            with mock.patch("tensorboardX.event_file_writer.time.time", return_value=1700000000.123):
                w1 = EventsWriter(prefix, filename_suffix=".test")
                w2 = EventsWriter(prefix, filename_suffix=".test")

                ev1 = event_pb2.Event(
                    summary=Summary(value=[Summary.Value(tag="first", simple_value=1.0)])
                )
                ev2 = event_pb2.Event(
                    summary=Summary(value=[Summary.Value(tag="second", simple_value=2.0)])
                )
                w1.write_event(ev1)
                w2.write_event(ev2)
                w1.close()
                w2.close()

            self.assertNotEqual(w1._file_name, w2._file_name)
            self.assertTrue(w1._file_name.endswith(".test"))
            self.assertTrue(w2._file_name.endswith(".test"))

            events1 = _read_tfrecord_events(w1._file_name)
            events2 = _read_tfrecord_events(w2._file_name)
            self.assertEqual([e.summary.value[0].tag for e in events1 if e.HasField("summary")], ["first"])
            self.assertEqual([e.summary.value[0].tag for e in events2 if e.HasField("summary")], ["second"])

    def test_concurrent_same_second_events_writers_use_distinct_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "events")
            barrier = threading.Barrier(4)
            names = []
            lock = threading.Lock()

            def create_writer():
                barrier.wait()
                w = EventsWriter(prefix)
                with lock:
                    names.append(w._file_name)
                w.close()

            with mock.patch("tensorboardX.event_file_writer.time.time", return_value=1700000000.5):
                threads = [threading.Thread(target=create_writer) for _ in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            self.assertEqual(len(names), 4)
            self.assertEqual(len(set(names)), 4)
