from tensorboardX import SummaryWriter
import unittest

from tensorboardX.record_writer import S3RecordWriter, make_valid_tf_name
class RecordWriterTest(unittest.TestCase):
    def test_record_writer_s3(self):
        writer = S3RecordWriter('s3://this/is/apen')

        bucket, path = writer.bucket_and_path()
        assert bucket == 'this'
        assert path == 'is/apen'

    def test_make_valid_tf_name(self):
        newname = make_valid_tf_name('$ave/&sound')
        assert newname == '._ave/_sound'