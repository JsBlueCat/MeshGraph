import torch
from torch.utils.cpp_extension import load
from torch_geometric.utils import to_undirected, remove_self_loops
# from models.layers.mesh_cpp_extension import Mesh
from models.layers.mesh import Mesh
import time


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def segregate_self_loops(edge_index, edge_attr=None):
    r"""Segregates self-loops from the graph.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`, :class:`LongTensor`,
        :class:`Tensor`)
    """

    mask = edge_index[0] != edge_index[1]
    inv_mask = ~mask

    loop_edge_index = edge_index[:, inv_mask]
    loop_edge_attr = None if edge_attr is None else edge_attr[inv_mask]
    edge_index = edge_index[:, mask]
    edge_attr = None if edge_attr is None else edge_attr[mask]

    return edge_index, edge_attr, loop_edge_index, loop_edge_attr


def remove_isolated_nodes(edge_index, edge_attr=None, num_nodes=None):
    r"""Removes the isolated nodes from the graph given by :attr:`edge_index`
    with optional edge attributes :attr:`edge_attr`.
    In addition, returns a mask of shape :obj:`[num_nodes]` to manually filter
    out isolated node features later on.
    Self-loops are preserved for non-isolated nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (LongTensor, Tensor, ByteTensor)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    out = segregate_self_loops(edge_index, edge_attr)
    edge_index, edge_attr, loop_edge_index, loop_edge_attr = out

    mask = torch.zeros(num_nodes, dtype=torch.uint8, device=edge_index.device)
    mask[edge_index.view(-1)] = 1

    assoc = torch.full((num_nodes, ), -1, dtype=torch.long, device=mask.device)
    assoc[mask] = torch.arange(mask.sum(), device=assoc.device)
    edge_index = assoc[edge_index]

    loop_mask = torch.zeros_like(mask)
    loop_mask[loop_edge_index[0]] = 1
    loop_mask = loop_mask & mask
    loop_assoc = torch.full_like(assoc, -1)
    loop_assoc[loop_edge_index[0]] = torch.arange(
        loop_edge_index.size(1), device=loop_assoc.device)
    loop_idx = loop_assoc[loop_mask]
    loop_edge_index = assoc[loop_edge_index[:, loop_idx]]

    edge_index = torch.cat([edge_index, loop_edge_index], dim=1)

    if edge_attr is not None:
        loop_edge_attr = loop_edge_attr[loop_idx]
        edge_attr = torch.cat([edge_attr, loop_edge_attr], dim=0)

    return edge_index, edge_attr, mask

class FaceToGraph(object):
    r"""Converts mesh faces :obj:`[3, num_faces]` to graph.

    Args:
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed.
    """

    def __init__(self, remove_faces=True):
        # self.mesh_graph_cpp = load(name='meshgraph_cpp', sources=[
        #     'models/layers/meshgraph.cpp'])
        self.remove_faces = remove_faces
        self.count = 0

    def __call__(self, data):
        start_time = time.time()
        data.num_nodes = data.x.size(0)
        edge_index = to_undirected(data.edge_index, data.num_nodes)
        # edge_index, _ = remove_self_loops(edge_index)

        print('%d-th mesh,size: %d' % (self.count, data.x.size(1)))
        
        data.edge_index = edge_index
        end_time = time.time()
        print('take {} s time for translate'.format(end_time-start_time))
        if self.remove_faces:
            data.face = None
        self.count += 1
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)



# class FaceToGraph(object):
#     r"""Converts mesh faces :obj:`[3, num_faces]` to graph.

#     Args:
#         remove_faces (bool, optional): If set to :obj:`False`, the face tensor
#             will not be removed.
#     """

#     def __init__(self, remove_faces=True):
#         # self.mesh_graph_cpp = load(name='meshgraph_cpp', sources=[
#         #     'models/layers/meshgraph.cpp'])
#         self.remove_faces = remove_faces
#         self.count = 0

#     def __call__(self, data):
#         start_time = time.time()
#         print('start transform')
#         mesh_grap = Mesh(data.pos, data.face)
#         # set the center ox oy oz unit_norm
#         data.x = mesh_grap.nodes
#         print(data.x)
#         data.num_nodes = data.x.size(0)
#         edge_index = to_undirected(mesh_grap.edge_index.t(), data.num_nodes)
#         # edge_index, _ = remove_self_loops(edge_index)

#         print('%d-th mesh,size: %d' % (self.count, data.x.size(0)))
#         # set edge_index  to data
#         data.edge_index = edge_index
#         end_time = time.time()
#         print('take {} s time for translate'.format(end_time-start_time))
#         mesh_grap = None
#         if self.remove_faces:
#             data.face = None
#         self.count += 1
#         return data

#     def __repr__(self):
#         return '{}()'.format(self.__class__.__name__)


class FaceToEdge(object):
    r"""Converts mesh faces :obj:`[3, num_faces]` to edge indices
    :obj:`[2, num_edges]`.

    Args:
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed.
    """

    def __init__(self, remove_faces=True):
        self.remove_faces = remove_faces
        self.count = 0

    def __call__(self, data):
        print(self.count)
        face = data.face

        edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
        edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

        data.edge_index = edge_index
        if self.remove_faces:
            data.face = None
        self.count += 1
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
