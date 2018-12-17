from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.versions_pb2 import VersionDef
from .proto.attr_value_pb2 import AttrValue
from .proto.tensor_shape_pb2 import TensorShapeProto

from collections import defaultdict
from typing import List
from typing import Dict

def AttrValue_proto(type,
                    shape,
                    s,
                    ):
    return


def Node_proto(name,
               op='UnSpecified',
               input=[],
               dtype=None,
               shape=None, # type: tuple
               outputsize=None,
               ):
    if not isinstance(input, list):
        input = [input]
    return NodeDef(
        name=name.encode(encoding='utf_8'),
        op=op,
        input=input,
    )



