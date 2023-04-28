from __future__ import absolute_import, division, print_function, unicode_literals
import importlib
import os
import sys

import google.protobuf.text_format as text_format
from google.protobuf.message import Message


def removeWhiteChar(string):
    return string.replace(' ', '').replace('\t', '').replace('\n', '')


def compare_proto(str_to_compare, function_ptr):
    module_id = function_ptr.__class__.__module__
    functionName = function_ptr.id().split('.')[-1]
    test_file = os.path.realpath(sys.modules[module_id].__file__)
    expected_file = os.path.join(os.path.dirname(test_file),
                        "expect",
                        module_id.split('.')[-1] + '.' + functionName + ".expect")
    print("expected_file: %s" % expected_file)
    assert os.path.exists(expected_file)
    with open(expected_file) as f:
        expected = f.read()

    if isinstance(str_to_compare, Message):

        proto_msg_module_name = str_to_compare.__class__.__module__
        proto_msg_class_name = str_to_compare.__class__.__name__

        proto_msg_module = importlib.import_module(proto_msg_module_name)
        ProtoMessage = getattr(proto_msg_module, proto_msg_class_name)

        expected_proto = ProtoMessage()
        text_format.Parse(expected, expected_proto)

        assert expected_proto == str_to_compare

    else:
        # TODO refactor tests to not compare tuple of protobuf messages in string
        # representation but protobuf messages themselves
        str_to_compare = str(str_to_compare)
        print("str_to_compare:", removeWhiteChar(str_to_compare))
        print("expected:", removeWhiteChar(expected))
        assert removeWhiteChar(str_to_compare) == removeWhiteChar(expected)


def write_proto(str_to_compare, function_ptr):
    module_id = function_ptr.__class__.__module__
    functionName = function_ptr.id().split('.')[-1]
    test_file = os.path.realpath(sys.modules[module_id].__file__)
    expected_file = os.path.join(os.path.dirname(test_file),
                    "expect",
                    module_id.split('.')[-1] + '.' + functionName + ".expect")
    print(expected_file)
    with open(expected_file, 'w') as f:
        f.write(str(str_to_compare))
