# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorboardX/proto/tensor.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorboardX.proto import resource_handle_pb2 as tensorboardX_dot_proto_dot_resource__handle__pb2
from tensorboardX.proto import tensor_shape_pb2 as tensorboardX_dot_proto_dot_tensor__shape__pb2
from tensorboardX.proto import types_pb2 as tensorboardX_dot_proto_dot_types__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1ftensorboardX/proto/tensor.proto\x12\x0ctensorboardX\x1a(tensorboardX/proto/resource_handle.proto\x1a%tensorboardX/proto/tensor_shape.proto\x1a\x1etensorboardX/proto/types.proto\"\xa9\x03\n\x0bTensorProto\x12%\n\x05\x64type\x18\x01 \x01(\x0e\x32\x16.tensorboardX.DataType\x12\x34\n\x0ctensor_shape\x18\x02 \x01(\x0b\x32\x1e.tensorboardX.TensorShapeProto\x12\x16\n\x0eversion_number\x18\x03 \x01(\x05\x12\x16\n\x0etensor_content\x18\x04 \x01(\x0c\x12\x14\n\x08half_val\x18\r \x03(\x05\x42\x02\x10\x01\x12\x15\n\tfloat_val\x18\x05 \x03(\x02\x42\x02\x10\x01\x12\x16\n\ndouble_val\x18\x06 \x03(\x01\x42\x02\x10\x01\x12\x13\n\x07int_val\x18\x07 \x03(\x05\x42\x02\x10\x01\x12\x12\n\nstring_val\x18\x08 \x03(\x0c\x12\x18\n\x0cscomplex_val\x18\t \x03(\x02\x42\x02\x10\x01\x12\x15\n\tint64_val\x18\n \x03(\x03\x42\x02\x10\x01\x12\x14\n\x08\x62ool_val\x18\x0b \x03(\x08\x42\x02\x10\x01\x12\x18\n\x0c\x64\x63omplex_val\x18\x0c \x03(\x01\x42\x02\x10\x01\x12>\n\x13resource_handle_val\x18\x0e \x03(\x0b\x32!.tensorboardX.ResourceHandleProtoB-\n\x18org.tensorflow.frameworkB\x0cTensorProtosP\x01\xf8\x01\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorboardX.proto.tensor_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\030org.tensorflow.frameworkB\014TensorProtosP\001\370\001\001'
  _TENSORPROTO.fields_by_name['half_val']._options = None
  _TENSORPROTO.fields_by_name['half_val']._serialized_options = b'\020\001'
  _TENSORPROTO.fields_by_name['float_val']._options = None
  _TENSORPROTO.fields_by_name['float_val']._serialized_options = b'\020\001'
  _TENSORPROTO.fields_by_name['double_val']._options = None
  _TENSORPROTO.fields_by_name['double_val']._serialized_options = b'\020\001'
  _TENSORPROTO.fields_by_name['int_val']._options = None
  _TENSORPROTO.fields_by_name['int_val']._serialized_options = b'\020\001'
  _TENSORPROTO.fields_by_name['scomplex_val']._options = None
  _TENSORPROTO.fields_by_name['scomplex_val']._serialized_options = b'\020\001'
  _TENSORPROTO.fields_by_name['int64_val']._options = None
  _TENSORPROTO.fields_by_name['int64_val']._serialized_options = b'\020\001'
  _TENSORPROTO.fields_by_name['bool_val']._options = None
  _TENSORPROTO.fields_by_name['bool_val']._serialized_options = b'\020\001'
  _TENSORPROTO.fields_by_name['dcomplex_val']._options = None
  _TENSORPROTO.fields_by_name['dcomplex_val']._serialized_options = b'\020\001'
  _globals['_TENSORPROTO']._serialized_start=163
  _globals['_TENSORPROTO']._serialized_end=588
# @@protoc_insertion_point(module_scope)
