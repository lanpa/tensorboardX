from .proto.attr_value_pb2 import AttrValue
from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.tensor_shape_pb2 import TensorShapeProto
from .proto.versions_pb2 import VersionDef


def load_openvino_graph(fname):
    nodes = []

    import xml.etree.ElementTree as ET
    tree = ET.parse(fname)
    root = tree.getroot()
    layers = root.find('layers')
    edges = root.find('edges')
    layers_dict = {}
    for layer in layers:
        nodeid = layer.attrib['id']
        name = layer.attrib['name']
        layers_dict[nodeid] = name
    for edge in edges:
        nodeinput = edge.attrib['from-layer']
        nodeself = edge.attrib['to-layer']
        attr = []
        # for s in node.attribute:
        #     attr.append(' = '.join([str(f[1]) for f in s.ListFields()]))
        attr = ', '.join(attr).encode(encoding='utf_8')
        nodes.append(NodeDef(
            name=layers_dict[nodeself],
            op='op',
            input=[str(layers_dict[nodeinput])],
            attr={'parameters': AttrValue(s=attr)},
        ))
    return GraphDef(node=nodes, versions=VersionDef(producer=22))
