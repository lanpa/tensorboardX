from tensorboardX import SummaryWriter, SummaryReader
import unittest
from pathlib import Path


class SummaryReaderTest(unittest.TestCase):
    def get_temp_dir(self):
        import tempfile
        return Path(tempfile.mkdtemp())

    def test_empty_record(self):
        tmpdir = self.get_temp_dir()
        with SummaryWriter(log_dir=tmpdir, filename_suffix='.tfevent') as writer:
            writer.flush()
        filename, = tmpdir.glob('*.tfevent')

        reader = SummaryReader(filename)
        events = list(reader)
        self.assertEqual(len(events), 1)
        event, = events
        assert event.HasField('file_version')
        self.assertEqual(list(event.summary.value), [])

    def test_scalar_record(self):
        tmpdir = self.get_temp_dir()
        n_events = 10
        with SummaryWriter(log_dir=tmpdir, filename_suffix='.tfevent') as writer:
            for n in range(n_events):
                writer.add_scalar('tag', n)
        filename, = tmpdir.glob('*.tfevent')

        reader = SummaryReader(filename)
        events = list(reader)
        self.assertEqual(len(events), n_events + 1)
        _, *events = events

        for n, event in enumerate(events):
            self.assertEqual(len(event.summary.value), 1)
            value, = event.summary.value
            self.assertEqual(value.tag, 'tag')
            self.assertEqual(value.simple_value, float(n))


if __name__ == '__main__':
    unittest.main()
