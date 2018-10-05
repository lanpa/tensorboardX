try:
    from . import event_pb2
except ImportError:
    raise RuntimeError('Run "./compile.sh" to compile protobuf bindings.')
