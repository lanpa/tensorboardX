# the file name is intended. pytest don't play well with multiprocessing

from tensorboardX import GlobalSummaryWriter as SummaryWriter
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import PyRecordReader_New
from tensorboardX.proto import event_pb2
import multiprocessing as mp
import numpy as np
import pytest
import unittest
import time

mp.set_start_method('fork')

class GlobalWriterTest(unittest.TestCase):
    def test_flush(self):
        N_TEST = 5
        w = SummaryWriter(flush_secs=1)
        f = w.file_writer.event_writer._ev_writer._file_name
        for i in range(N_TEST):
            w.add_scalar('a', i)
            time.sleep(2)
        r = PyRecordReader_New(f)
        r.GetNext()  # meta data, so skip
        for _ in range(N_TEST):  # all of the data should be flushed
            r.GetNext()

    def test_flush_timer_is_long_so_data_is_not_there(self):
        with self.assertRaises(BaseException):
            N_TEST = 5
            w = SummaryWriter(flush_secs=20)
            f = w.file_writer.event_writer._ev_writer._file_name
            for i in range(N_TEST):
                w.add_scalar('a', i)
                time.sleep(2)
            r = PyRecordReader_New(f)
            r.GetNext()  # meta data, so skip
            for _ in range(N_TEST):  # missing data
                r.GetNext()

    def test_flush_after_close(self):
        N_TEST = 5
        w = SummaryWriter(flush_secs=20)
        f = w.file_writer.event_writer._ev_writer._file_name
        for i in range(N_TEST):
            w.add_scalar('a', i)
            time.sleep(2)
        w.close()
        r = PyRecordReader_New(f)
        r.GetNext()  # meta data, so skip
        for _ in range(N_TEST):  # all of the data should be flushed
            r.GetNext()


    def test_auto_close(self):
        pass

    def test_writer(self):
        TEST_LEN = 100
        N_PROC = 4
        writer = SummaryWriter()
        event_filename = writer.file_writer.event_writer._ev_writer._file_name

        predifined_values = list(range(TEST_LEN))
        def train3():
            for i in range(TEST_LEN):
                writer.add_scalar('many_write_in_func', predifined_values[i])
                time.sleep(0.01*np.random.randint(0, 10))

        processes = []
        for i in range(N_PROC):
            p1 = mp.Process(target=train3)
            processes.append(p1)
            p1.start()

        for p in processes:
            p.join()
        writer.close()


        collected_values = []
        r = PyRecordReader_New(event_filename)
        r.GetNext()  # meta data, so skip
        for _ in range(TEST_LEN*N_PROC):  # all of the data should be flushed
            r.GetNext()
            ev = event_pb2.Event()
            value = ev.FromString(r.record()).summary.value
            collected_values.append(value[0].simple_value)

        collected_values = sorted(collected_values)
        for i in range(TEST_LEN):
            for j in range(N_PROC):
                assert collected_values[i*N_PROC+j] == i

