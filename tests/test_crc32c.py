import unittest
from secrets import token_bytes
from tensorboardX.crc32c import _crc32c, _crc32c_native


class CRC32CTest(unittest.TestCase):
    def test_crc32c(self):
        data = b'abcd'
        assert _crc32c(data) == 0x92c80a31

    def test_implementations(self):
        random_data = token_bytes(100)
        a = _crc32c(random_data)
        b = _crc32c_native(random_data)
        assert a == b
