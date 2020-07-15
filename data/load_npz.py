import numpy as np
import torch
from torch_geometric.data import Data


def read_npz(path, train):
    with np.load(path) as f:
        return parse_npz(f, train)


def parse_npz(f, train):
    face = f['face']
    neighbor_index = f['neighbor_index']

    # data augmentation
    if train:
        sigma, clip = 0.01, 0.05
        jittered_data = np.clip(
            sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
        face = np.concatenate((face[:, :12] + jittered_data, face[:, 12:]), 1)

    # fill for n < max_faces with randomly picked faces
    num_point = len(face)
    if num_point < 1024:
        fill_face = []
        fill_neighbor_index = []
        for i in range(1024 - num_point):
            index = np.random.randint(0, num_point)
            fill_face.append(face[index])
            fill_neighbor_index.append(neighbor_index[index])
        face = np.concatenate((face, np.array(fill_face)))
        neighbor_index = np.concatenate(
            (neighbor_index, np.array(fill_neighbor_index)))

    # to tensor
    face = torch.from_numpy(face).float()
    neighbor_index = torch.from_numpy(neighbor_index).long()
    index = torch.arange(face.size(0)).unsqueeze(dim=1).repeat(1, 3)
    gather_index = torch.tensor([0, 3, 1, 4, 2, 5]).repeat(face.size(0), 1)
    edge_index = torch.cat([neighbor_index, index], dim=1).gather(
        1, gather_index).view(-1, 2).permute(1, 0)

    
    # reorganize
    face = face.permute(1, 0)
    centers, corners, normals = face[:3], face[3:12], face[12:]


    # # get the sod of each faces
    # '''
    # w(e) = cos(ni/||ni|| *  nj/||nj||)^-1
    # '''
    # start_point, end_point = edge_index[0, :], edge_index[1, :]

    # print(start_point.size())
    # print(normals.size())




    corners = corners - torch.cat([centers, centers, centers], 0)

    features = torch.cat([centers, corners, normals], dim=0).permute(1, 0)

    data = Data(x=features, edge_index=edge_index, pos=neighbor_index)
    return data
