from tensorboardX import SummaryWriter
import tempfile
import unittest


class SummaryWriterTest(unittest.TestCase):
    def test_summary_writer_ctx(self):
        # after using a SummaryWriter as a ctx it should be closed
        with SummaryWriter(filename_suffix='.test') as writer:
            writer.add_scalar('test', 1)
        assert writer.file_writer is None

    def test_summary_writer_backcompat(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with SummaryWriter(log_dir=tmp_dir) as writer:
                writer.add_scalar('test', 1)

    def test_summary_writer_close(self):
        # Opening and closing SummaryWriter a lot should not run into
        # OSError: [Errno 24] Too many open files
        for i in range(1000):
            writer = SummaryWriter()
            writer.close()

    def test_windowsPath(self):
        dummyPath = "C:\\Downloads\\fjoweifj02utj43tj430"
        with SummaryWriter(dummyPath) as writer:
            writer.add_scalar('test', 1)
        import shutil
        shutil.rmtree(dummyPath)

    def test_pathlib(self):
        import sys
        if sys.version_info.major == 2:
            import pathlib2 as pathlib
        else:
            import pathlib
        p = pathlib.Path('./pathlibtest')
        with SummaryWriter(p) as writer:
            writer.add_scalar('test', 1)
        import shutil
        shutil.rmtree(str(p))

    def test_dummy_summary_writer(self):
        # You can't write to root folder without sudo.
        with SummaryWriter('/', write_to_disk=False) as writer:
            writer.add_scalar('test', 1)
            writer.flush()