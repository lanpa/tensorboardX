from tensorboardX import SummaryWriter
import unittest


class SummaryWriterTest(unittest.TestCase):
    def test_summary_writer_ctx(self):
        # after using a SummaryWriter as a ctx it should be closed
        with SummaryWriter(filename_suffix='.test') as writer:
            writer.add_scalar('test', 1)
        assert writer.file_writer is None


    def test_summary_writer_close(self):
        # Opening and closing SummaryWriter a lot should not run into
        # OSError: [Errno 24] Too many open files
        passed = True
        try:
            writer = SummaryWriter()
            writer.close()
        except OSError:
            passed = False

        assert passed


    def test_windowsPath(self):
        with SummaryWriter("C:\\Downloads") as writer:
            writer.add_scalar('test', 1)


    def test_pathlib(self):
        import pathlib
        p = pathlib.Path('./pathlibtest')
        with SummaryWriter(p) as writer:
            writer.add_scalar('test', 1)
