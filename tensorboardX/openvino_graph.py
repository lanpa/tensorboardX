from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.versions_pb2 import VersionDef
from .proto.attr_value_pb2 import AttrValue
from .proto.tensor_shape_pb2 import TensorShapeProto


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

    g = m.graph
    return parse(g)


def parse(graph):
    nodes_proto = []
    nodes = []
    import itertools
    for node in itertools.chain(graph.input, graph.output):
        nodes_proto.append(node)

    for node in nodes_proto:
        print(node.name)
        shapeproto = TensorShapeProto(
            dim=[TensorShapeProto.Dim(size=d.dim_value) for d in node.type.tensor_type.shape.dim])
        nodes.append(NodeDef(
            name=node.name.encode(encoding='utf_8'),
            op='Variable',
            input=[],
            attr={
                'dtype': AttrValue(type=node.type.tensor_type.elem_type),
                'shape': AttrValue(shape=shapeproto),
            })
        )

    for node in graph.node:
        attr = []
        for s in node.attribute:
            attr.append(' = '.join([str(f[1]) for f in s.ListFields()]))
        attr = ', '.join(attr).encode(encoding='utf_8')
        print(node.output[0])
        nodes.append(NodeDef(
            name=node.output[0].encode(encoding='utf_8'),
            op=node.op_type,
            input=node.input,
            attr={'parameters': AttrValue(s=attr)},
        ))

    # two pass token replacement, appends opname to object id
    mapping = {}
    for node in nodes:
        mapping[node.name] = node.op + '_' + node.name

    return GraphDef(node=nodes, versions=VersionDef(producer=22))
