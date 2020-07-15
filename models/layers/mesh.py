import torch
import numpy as np
import math


class Mesh:
    '''
    mesh graph obj contains pos edge_index and x
    x reprensent feature in face obj
    edge_index is the sparse matrix of adj matrix
    pos represent position matrix
    '''

    def __init__(self, vertexes, faces):
        self.vertexes = vertexes
        self.faces = faces.t().long()
        self.edge_index = self.nodes = None
        self.num_nodes = self.vertexes.size(0)
        # find point and face indices
        self.sorted_point_to_face = None
        self.sorted_point_to_face_index_dict = None
        # normalize vertexes
        self.normlize_vertices()

        # create graph
        self.create_graph()

    def create_graph(self):
        # conat center ox oy oz norm
        pos = self.vertexes[self.faces]
        point_x, point_y, point_z = pos[:, 0, :], pos[:, 1, :], pos[:, 2, :]
        centers = get_inner_center_vec(
            point_x, point_y, point_z
        )
        temp = centers.view(-1)
        if torch.sum(torch.isnan(temp), dim=0) > 0:
            raise('center -------------------------------  nan')
        ox, oy, oz = get_three_vec(centers, point_x, point_y, point_z)
        norm = get_unit_norm_vec(point_x, point_y, point_z)
        temp = norm.view(-1)
        if torch.sum(torch.isnan(temp), dim=0) > 0:
            raise('norm -------------------------------   nan')
        # cat the vecter
        self.nodes = torch.cat((centers, ox, oy, oz, norm), dim=1)

        is_nan = self.nodes.view(-1)
        if torch.sum(torch.isnan(is_nan), dim=0) > 0:
            raise('contain nan ')
        self.get_connect_matrix()

    def normlize_vertices(self):
        ''' move vertices to center
        '''
        center = (torch.max(self.vertexes, dim=0)[0] +
                  torch.min(self.vertexes, dim=0)[0])/2
        self.vertexes -= center
        max_len = torch.max(self.vertexes[:, 0]**2 +
                            self.vertexes[:, 1]**2 + self.vertexes[:, 2]**2).item()
        self.vertexes /= math.sqrt(max_len)

    def get_connect_matrix(self):
        node_list = [[] for _ in range(self.num_nodes)]
        result = []
        for i, v in enumerate(self.faces):
            v0, v1, v2 = v
            node_list[v0].append(i)
            node_list[v1].append(i)
            node_list[v2].append(i)
        for i in node_list:
            for p in range(0, len(i)-1):
                for q in range(p+1, len(i)):
                    result.append([p, q])
        self_loop = torch.arange(self.nodes.size(0)).repeat(2, 1).t().long()
        result = torch.tensor(result).long()
        self.edge_index = torch.cat([self_loop, result], dim=0)


def get_inner_center_vec(v1, v2, v3):
    '''
    v1 v2 v3 represent 3 vertexes of triangle
    v1 (n,3)
    '''
    return (v1+v2+v3)/3


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
    return norm


def get_three_vec(center, v1, v2, v3):
    '''
    return ox oy oz vector
    '''
    return v1-center, v2-center, v3-center
