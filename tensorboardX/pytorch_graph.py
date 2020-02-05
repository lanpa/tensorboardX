import logging
import time
from collections import OrderedDict
from .proto.attr_value_pb2 import AttrValue
from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.step_stats_pb2 import RunMetadata, StepStats, DeviceStepStats, NodeExecStats, AllocatorMemoryUsed
from .proto.tensor_shape_pb2 import TensorShapeProto
from .proto.versions_pb2 import VersionDef
from .proto_graph import node_proto

methods_OP = ['attributeNames', 'hasMultipleOutputs', 'hasUses', 'inputs',
              'kind', 'outputs', 'outputsSize', 'scopeName']
methods_IO = []

GETATTR_KIND = 'prim::GetAttr'
CLASSTYPE_KIND = 'ClassType'
CONST_KIND = 'prim::Constant'


class NodeBase(object):
    def __init__(self,
                 debugName=None,
                 inputs=None,
                 scope=None,
                 tensor_size=None,
                 op_type='UnSpecified',
                 attributes=''):
        self.debugName = debugName
        self.inputs = inputs
        self.tensor_size = tensor_size
        self.kind = op_type
        self.attributes = attributes
        if scope is not None:
            self.scope = scope

    def __repr__(self):
        repr = []
        repr.append(str(type(self)))
        for m in dir(self):
            if '__' not in m:
                repr.append(m + ': ' + str(getattr(self, m)) + str(type(getattr(self, m))))
        return '\n'.join(repr) + '\n\n'


class NodePy(NodeBase):
    def __init__(self, node_cpp, valid_methods):
        super(NodePy, self).__init__(node_cpp)
        valid_methods = valid_methods[:]
        self.inputs = []
        for m in valid_methods:
            if m == 'inputs' or m == 'outputs':
                list_of_node = list(getattr(node_cpp, m)())
                io_unique_names = []
                io_tensor_sizes = []
                for n in list_of_node:
                    io_unique_names.append(n.debugName())

                    if n.isCompleteTensor():
                        io_tensor_sizes.append(n.type().sizes())
                    else:
                        io_tensor_sizes.append(None)

                setattr(self, m, io_unique_names)
                setattr(self, m + 'tensor_size', io_tensor_sizes)

            else:
                setattr(self, m, getattr(node_cpp, m)())


class NodePyIO(NodePy):
    def __init__(self, node_cpp, input_or_output=None, debugName='', tensor_size=[]):
        super(NodePyIO, self).__init__(node_cpp, methods_IO)
        self.tensor_size = tensor_size
        # Kind attribute string is purely descriptive and will be shown
        # in detailed information for the node in TensorBoard's graph plugin.
        #
        # NodePyOP nodes get this from their kind() method.
        self.debugName = debugName
        self.kind = 'Parameter'
        if input_or_output:
            self.input_or_output = input_or_output
            self.kind = 'IO Node'


class NodePyOP(NodePy):
    def __init__(self, node_cpp):
        super(NodePyOP, self).__init__(node_cpp, methods_OP)
        # Replace single quote which causes strange behavior in TensorBoard
        # TODO: See if we can remove this in the future
        self.attributes = str({k: node_cpp[k] for k in node_cpp.attributeNames()}).replace("'", ' ')
        self.kind = node_cpp.kind()


class GraphPy(object):
    """Helper class to convert torch.nn.Module to GraphDef proto and visualization
    with TensorBoard.

    GraphDef generation operates in two passes:

    In the first pass, all nodes are read and saved to two lists.
    One list is for input/output nodes (nodes_io), which only have inbound
    or outbound connections, but not both. Another list is for internal
    operator nodes (nodes_op). The first pass also saves all scope name
    appeared in the nodes in scope_name_appeared list for later processing.

    In the second pass, scope names are fully applied to all nodes.
    debugNameToScopedName is a mapping from a node's ID to its fully qualified
    scope name. e.g. Net1/Linear[0]/1. Unfortunately torch.jit doesn't have
    totally correct scope output, so this is nontrivial. The function
    populate_namespace_from_OP_to_IO and find_common_root are used to
    assign scope name to a node based on the connection between nodes
    in a heuristic kind of way. Bookkeeping is done with shallowest_scope_name
    and scope_name_appeared.
    """
    def __init__(self):
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.unique_name_to_scoped_name = {}
        self.shallowest_scope_name = 'default'
        self.scope_name_appeared = []
        self.profile_result = None

    def append(self, x):
        if isinstance(x, NodePyIO):
            self.nodes_io[x.debugName] = x
        if isinstance(x, NodePyOP):
            self.nodes_op.append(x)
            for node_output, outputSize in zip(x.outputs, x.outputstensor_size):
                self.scope_name_appeared.append(x.scopeName)
                self.nodes_io[node_output] = NodeBase(node_output,
                                                      x.inputs,
                                                      x.scopeName,
                                                      outputSize,
                                                      op_type=x.kind,
                                                      attributes=x.attributes)

    def printall(self):
        print('all nodes')
        for node in self.nodes_op:
            print(node)
        for key in self.nodes_io:
            print(self.nodes_io[key])

    def find_common_root(self):
        """
        Find the shallowest scope name among the appeared nodes.
        """
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowest_scope_name = fullscope.split('/')[0]

    def populate_namespace_from_OP_to_IO(self):
        for node in self.nodes_op:
            for node_output, outputSize in zip(node.outputs, node.outputstensor_size):
                self.scope_name_appeared.append(node.scopeName)
                self.nodes_io[node_output] = NodeBase(node_output,
                                                      node.inputs,
                                                      node.scopeName,
                                                      outputSize,
                                                      op_type=node.kind,
                                                      attributes=node.attributes)

        self.find_common_root()

        for node in self.nodes_op:
            for input_node_id in node.inputs:
                self.unique_name_to_scoped_name[input_node_id] = node.scopeName + '/' + input_node_id

        for key, node in self.nodes_io.items():
            if type(node) == NodeBase:
                self.unique_name_to_scoped_name[key] = node.scope + '/' + node.debugName
            if hasattr(node, 'input_or_output'):
                self.unique_name_to_scoped_name[key] = node.input_or_output + '/' + node.debugName
            if hasattr(node, 'scope') and node.scope is not None:
                self.unique_name_to_scoped_name[key] = node.scope + '/' + node.debugName
                if node.scope == '' and self.shallowest_scope_name:
                    self.unique_name_to_scoped_name[node.debugName] = \
                        self.shallowest_scope_name + '/' + node.debugName

        # replace name
        for key, node in self.nodes_io.items():
            self.nodes_io[key].inputs = \
                [self.unique_name_to_scoped_name[node_input_id] for node_input_id in node.inputs]
            if node.debugName in self.unique_name_to_scoped_name:
                self.nodes_io[key].debugName = self.unique_name_to_scoped_name[node.debugName]

    def to_proto(self):
        """
        Converts graph representation of GraphPy object to TensorBoard
        required format.
        """
        # TODO: compute correct memory usage and CPU time once
        # PyTorch supports it
        import numpy as np
        nodes = []
        node_stats = []

        if self.profile_result is not None:
            profile_result = self.profile_result.function_events

        _time_used_for_op = {}

        # We assume that the model is executed sequentially. So get the timing from
        # the first matched item. If it is matched, remove that item with `pop()`
        def find_time_for(node_name):
            for i, n in enumerate(profile_result):
                if n.key == node_name:
                    profile_result.pop(i)
                    time_we_want_cpu = n.cpu_time_total
                    time_we_want_cuda = n.cuda_time_total

                    return int(time_we_want_cpu), int(time_we_want_cuda)
            return None, None

        should_show_warning = False
        for v in self.nodes_io.values():
            nodes.append(node_proto(v.debugName,
                                    input=v.inputs,
                                    outputsize=v.tensor_size,
                                    op=v.kind,
                                    attributes=v.attributes))

            # For timing information, we are only interested in aten operators now.
            # prim:: and Parameter
            if 'aten' in v.kind and self.profile_result is not None:
                opname = v.kind.split('::')[1]
                exe_time_cpu, exe_time_cuda = find_time_for(opname)
                if exe_time_cpu is not None:
                    total_time = exe_time_cpu + exe_time_cuda

                    # assume that the operation will not executed on both device simultaneously.
                    if total_time - max(exe_time_cpu, exe_time_cuda) > 0.01:
                        should_show_warning = True

                    node_stats.append(
                        NodeExecStats(node_name=v.debugName,
                                      all_start_micros=int(time.time() * 1e7),
                                      all_end_rel_micros=total_time))

            if v.tensor_size and len(v.tensor_size) > 0:  # assume data is float32, only parameter is counted
                node_stats.append(
                    NodeExecStats(node_name=v.debugName,
                                  all_start_micros=int(time.time() * 1e7),
                                  all_end_rel_micros=42,
                                  memory=[AllocatorMemoryUsed(allocator_name="unknown",
                                                              total_bytes=int(np.prod(v.tensor_size)) * 4)]))
        if should_show_warning:
            logging.warning('time cost for node is the sum of CPU + GPU.')

        return nodes, node_stats


# one argument: 'hasAttribute', 'hasAttributes',
def parse(graph, trace, args=None, profile_result=None):
    """This method parses an optimized PyTorch model graph and produces
    a list of nodes and node stats for eventual conversion to TensorBoard
    protobuf format.

    Args:
      graph (PyTorch module): The model graph to be parsed.
      trace (PyTorch JIT TracedModule): The model trace to be parsed.
      args (tuple): input tensor[s] for the model.
    """
    import torch
    n_inputs = len(args)  # not sure...

    inputnodes = list(graph.inputs())

    nodes_py = GraphPy()
    nodes_py.profile_result = profile_result

    for node in graph.inputs():
        if node.type().kind() == CLASSTYPE_KIND:
            continue
        try:
            tensor_size = node.type().sizes()
        except RuntimeError:
            # INTERNAL ASSERT FAILED at ../aten/src/ATen/core/jit_type.h:131, please report a bug to PyTorch.
            tensor_size = []
        nodes_py.append(NodePyIO(node, input_or_output='Input', debugName=node.debugName(), tensor_size=tensor_size))
    attr_to_scope = dict()
    for node in graph.nodes():
        # These nodes refers to parameters such as kernel size, stride, etc.
        # The graph will be very tedious if we include all of them. So skip.
        # p.s. Those Constant will be composed by 'prim::listConstruct' and then
        # send to common OPs such as Maxpool, Conv, Linear.
        # We can let user pass verbosity value to dicide how detailed the graph is.
        if node.kind() == CONST_KIND:
            continue
        if node.kind() == GETATTR_KIND:
            attr_name = node.s('name')
            parent = node.input().node()
            if parent.kind() == GETATTR_KIND:  # If the parent node is not the top-level "self" node
                parent_attr_name = parent.s('name')
                parent_scope = attr_to_scope[parent_attr_name]
                attr_scope = parent_scope.split('/')[-1]
                attr_to_scope[attr_name] = '{}/{}.{}'.format(parent_scope, attr_scope, attr_name)
            else:
                attr_to_scope[attr_name] = '__module.{}'.format(attr_name)
            # We don't need classtype nodes; scope will provide this information
            if node.output().type().kind() == CLASSTYPE_KIND:
                continue
            node_py = NodePyOP(node)
            node_py.scopeName = attr_to_scope[attr_name]
            nodes_py.append(node_py)
        else:
            nodes_py.append(NodePyOP(node))
    for i, node in enumerate(graph.outputs()):  # Create sink nodes for output ops
        if node.isCompleteTensor():
            node_py = NodePyIO(node, 'output')
            node_py.debugName = "output.{}.alias".format(node.debugName())
            node_py.inputs = [node.debugName()]
            nodes_py.append(node_py)
        else:  # tuple output (prim::TupleConstruct)
            graph_outputs = list(node.node().inputs())
            for go in graph_outputs:
                node_py = NodePyIO(go, 'output')
                node_py.debugName = "output.{}.alias".format(go.debugName())
                node_py.inputs = [go.debugName()]
                nodes_py.append(node_py)

    def parse_traced_name(module):
        if isinstance(module, torch.jit.TracedModule):
            module_name = module._name
        else:
            module_name = getattr(module, 'original_name', "Module")
        return module_name

    alias_to_name = dict()
    base_name = parse_traced_name(trace)
    for name, module in trace.named_modules(prefix='__module'):
        mod_name = parse_traced_name(module)
        attr_name = name.split('.')[-1]
        alias_to_name[name] = '{}[{}]'.format(mod_name, attr_name)
    for node in nodes_py.nodes_op:
        module_aliases = node.scopeName.split('/')
        replacements = [
            alias_to_name[alias]
            if alias in alias_to_name
            else alias.split('.')[-1]
            for alias in module_aliases
        ]
        node.scopeName = base_name
        if any(replacements):
            node.scopeName += '/' + '/'.join(replacements)

    nodes_py.populate_namespace_from_OP_to_IO()
    return nodes_py.to_proto()


def recursive_to_cuda(x):
    """
    Recursively convert tensors in a tuple or list to GPU tensor.
    """
    import torch

    if isinstance(x, torch.Tensor):
        return x.cuda()
    else:
        return [recursive_to_cuda(_x) for _x in x]


def graph(model, args, verbose=False, use_cuda=False, **kwargs):
    """
    This method processes a PyTorch model and produces a `GraphDef` proto
    that can be logged to TensorBoard.

    Args:
      model (PyTorch module): The model to be parsed.
      args (tuple): input tensor[s] for the model.
      verbose (bool): Whether to print out verbose information while
        processing.
    """
    import torch
    from packaging import version
    assert version.parse(torch.__version__) >= version.parse("1.4.0"), "add_graph needs torch>=1.4.0"

    with torch.onnx.set_training(model, False):  # TODO: move outside of torch.onnx
        try:
            trace = torch.jit.trace(model, args)
            if type(trace) == torch.jit.ScriptModule:
                graph = trace.forward_impl.graph
            else:
                graph = trace.graph
            torch._C._jit_pass_inline(graph)
        except RuntimeError as e:
            print(e)
            print('Error occurs, No graph saved')
            raise e
            # Create an object matching
            # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/graph.proto
            # The producer version has been reverse engineered from standard
            # TensorBoard logged data.

        try:
            if use_cuda:
                model.cuda()
                args = recursive_to_cuda(args)
            with torch.autograd.profiler.profile(record_shapes=True, use_cuda=use_cuda) as prof:
                result = model(*args)

        except RuntimeError as e:
            print('profiler execution failed')
            prof = None

    if verbose:
        print(graph)
    list_of_nodes, node_stats = parse(graph, trace, args, prof)
    # We are hardcoding that this was run on CPU even though it might have actually
    # run on GPU. Note this is what is shown in TensorBoard and has no bearing
    # on actual execution.
    # TODO: See if we can extract GPU vs CPU information from the PyTorch model
    # and pass it correctly to TensorBoard.
    #
    # Definition of StepStats and DeviceStepStats can be found at
    # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/graph/tf_graph_common/test/graph-test.ts
    # and
    # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/step_stats.proto

    if use_cuda:
        device = "/device:GPU:0"
    else:
        device = "/device:CPU:0"
    stepstats = RunMetadata(step_stats=StepStats(dev_stats=[DeviceStepStats(device=device,
                                                                            node_stats=node_stats)]))
    return GraphDef(node=list_of_nodes, versions=VersionDef(producer=22)), stepstats
