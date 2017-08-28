from .src.graph_pb2 import GraphDef
from .src.node_def_pb2 import NodeDef
from .src.versions_pb2 import VersionDef
from .src.attr_value_pb2 import AttrValue
from .src.tensor_shape_pb2 import TensorShapeProto

global id2name
global list_of_nodes
def make_name(obj):
    if hasattr(obj, 'variable'):#weight/bias in module
        if obj.variable is not None:
            return id2name[id(obj.variable)]+'_'+str(id(obj.variable))
        else:
            return 'inputTensor_'+str(id(obj))
    else:
        return type(obj).__name__.replace('Backward','_')+str(id(obj))

def make_list_of_nodes(fn):
    if fn is None:
        return 
    inputs = []
    for next_fn, _ in fn.next_functions:
        inputs.append(make_name(next_fn))
        make_list_of_nodes(next_fn)
    attrshape = []   
    if hasattr(fn, 'variable'):#weight/bias in module
        if fn.variable is not None:
            attrshape = list(fn.variable.size())
    list_of_nodes.append({'name':make_name(fn), 'op':type(fn).__name__, 'inputs':inputs, 'attr.shape':attrshape})



def graph(model, lastVar):
    global id2name
    global list_of_nodes
    id2name = {id(m):n.replace('.', '/')+'(parameters)' for n, m in model.named_parameters()}
    nodes = []
    list_of_nodes = []
    make_list_of_nodes(lastVar.grad_fn)
    for node in list_of_nodes:
        #shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=i) for i in node['attr.shape']])  ugly...
        shape_str = str(node['attr.shape']).encode(encoding='utf_8')
        nodes.append(NodeDef(name=node['name'], op=node['op'], input=node['inputs'], attr={'shape':AttrValue(s=shape_str)}))#, 'T':AttrValue(type="DT_FLOAT")}))
    return GraphDef(node=nodes, versions=VersionDef(producer=22))
