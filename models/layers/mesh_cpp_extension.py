import torch
import numpy as np


class Mesh:
    '''
    mesh graph obj contains pos edge_index and x
    x reprensent feature in face obj
    edge_index is the sparse matrix of adj matrix
    pos represent position matrix
    '''

    def __init__(self, vertexes, faces, meshgraph_cpp):
        self.vertexes = vertexes
        self.faces = faces.t()
        self.edge_index = self.nodes = None
        self.num_nodes = self.vertexes.size(0)
        self.meshgraph_cpp = meshgraph_cpp
        # find point and face indices
        self.sorted_point_to_face = None
        self.sorted_point_to_face_index_dict = None
        # create graph
        self.create_graph()

    def create_graph(self):
        # conat center ox oy oz norm
        pos = self.vertexes[self.faces]
        point_x, point_y, point_z = pos[:, 0, :], pos[:, 1, :], pos[:, 2, :]
        centers = get_inner_center_vec(
            point_x, point_y, point_z
        )
        ox, oy, oz = get_three_vec(centers, point_x, point_y, point_z)
        norm = get_unit_norm_vec(point_x, point_y, point_z)
        # cat the vecter
        self.nodes = torch.cat((centers, ox, oy, oz, norm), dim=1)
        self.edge_index = self.meshgraph_cpp.get_connect_matrix(
            self.faces, self.num_nodes
        )


def get_inner_center_vec(v1, v2, v3):
    '''
    v1 v2 v3 represent 3 vertexes of triangle
    v1 (n,3)
    '''
    a = get_distance_vec(v2, v3)
    b = get_distance_vec(v3, v1)
    c = get_distance_vec(v1, v2)
    x = torch.stack((v1[:, 0], v2[:, 0], v3[:, 0]), dim=1)
    y = torch.stack((v1[:, 1], v2[:, 1], v3[:, 1]), dim=1)
    z = torch.stack((v1[:, 2], v2[:, 2], v3[:, 2]), dim=1)
    dis = torch.stack((a, b, c), dim=1)
    return torch.stack((
        torch.sum((x * dis) / (a+b+c).repeat(3, 1).t(), dim=1),
        torch.sum((y * dis) / (a+b+c).repeat(3, 1).t(), dim=1),
        torch.sum((z * dis) / (a+b+c).repeat(3, 1).t(), dim=1),
    ), dim=1)


def get_distance_vec(v1, v2):
    '''
    get distance between
    vecter_1 and vecter_2
    v1 : (x,y,z)
    '''
    return torch.sqrt(torch.sum((v1-v2)**2, dim=1))


def get_unit_norm_vec(v1, v2, v3):
    '''
    xy X xz
    （y1z2-y2z1,x2z1-x1z2,x1y2-x2y1）
    '''
    xy = v2-v1
    xz = v3-v1
    x1, y1, z1 = xy[:, 0], xy[:, 1], xy[:, 2]
    x2, y2, z2 = xz[:, 0], xz[:, 1], xz[:, 2]
    norm = torch.stack((y1*z2-y2*z1, x2*z1-x1*z2, x1*y2-x2*y1), dim=1)
    vec_len = torch.sqrt(torch.sum(norm, dim=1))
    return norm / vec_len.repeat(3, 1).t()


def get_three_vec(center, v1, v2, v3):
    '''
    return ox oy oz vector
    '''
    return v1-center, v2-center, v3-center
