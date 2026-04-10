import unittest
import multiprocessing as mp
import os
import time
import numpy as np
from tensorboardX import SummaryWriter
from tensorboardX.proto import event_pb2

def worker_fn(w):
    # In 'spawn' mode, 'w' is a unpickled copy.
    # On Windows, the thread isn't running here, but add_scalar
    # should still put serialized bytes into the shared queue.
    try:
        w.add_scalar('worker_metric', 1.23, global_step=10)
    except Exception as e:
        print(f"Worker failed: {e}")

class Issue727Test(unittest.TestCase):
    def test_multiprocess_serialization(self):
        # Use 'spawn' to simulate Windows behavior if possible
        try:
            ctx = mp.get_context('spawn')
        except ValueError:
            ctx = mp.get_context('fork')
        
        writer = SummaryWriter()
        event_filename = writer.file_writer.event_writer._ev_writer._file_name
        
        p = ctx.Process(target=worker_fn, args=(writer,))
        p.start()
        p.join()
        
        writer.close()
        
        # Verify data was written correctly
        from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import PyRecordReader_New
        r = PyRecordReader_New(event_filename)
        r.GetNext()  # meta data
        r.GetNext()  # the metric
        
        ev = event_pb2.Event()
        ev.ParseFromString(r.record())
        self.assertEqual(ev.step, 10)
        self.assertEqual(ev.summary.value[0].tag, 'worker_metric')
        self.assertAlmostEqual(ev.summary.value[0].simple_value, 1.23)

    def test_atexit_pid_check(self):
        writer = SummaryWriter()
        original_pid = writer.file_writer._pid
        self.assertEqual(original_pid, os.getpid())
        self.assertTrue(hasattr(writer.file_writer, '_pid'))

if __name__ == '__main__':
    unittest.main()
