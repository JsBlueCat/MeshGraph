import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from models.layers.struct_conv import MeshMlp, NormMlp
from models.layers.mesh_net_with_out_neigbour import SpatialDescriptor, StructuralDescriptor, MeshConvolution
# ,SpatialDescriptor, StructuralDescriptor, MeshConvolution
from models.layers.mesh_graph_conv import NormGraphConv, GINConv


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, init_type, init_gain, cuda_ids):
    if len(cuda_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(cuda_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, cuda_ids)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def get_net(opt):
    net = None
    if opt.arch == 'meshconv':
        net = MeshGraph(opt)
    else:
        raise NotImplementedError(
            'model name [%s] is not implemented' % opt.arch)
    return init_net(net, opt.init_type, opt.init_gain, opt.cuda)


def get_loss(opt):
    if opt.task == 'cls':
        loss = nn.CrossEntropyLoss()
    elif opt.task == 'seg':
        loss = nn.CrossEntropyLoss(ignore_index=-1)
    return loss


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, gamma=opt.gamma, milestones=opt.milestones)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# class MeshGraph(nn.Module):
#     """Some Information about MeshGraph"""

#     def __init__(self, opt):
#         super(MeshGraph, self).__init__()
#         self.mesh_mlp_256 = MeshMlp(opt, 256)
#         self.gin_conv_256 = GINConv(self.mesh_mlp_256)

#         # self.graph_conv_64 = GraphConv(1024, 256)
#         # self.graph_conv_64 = GraphConv(256, 64)

#         self.classifier = nn.Sequential(
#             nn.Linear(256, 1024),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(256, 40),
#         )

#         if opt.use_fpm:
#             self.mesh_mlp_64 = MeshMlp(64)
#             self.mesh_mlp_128 = MeshMlp(128)
#             self.gin_conv_64 = GINConv(self.mesh_mlp_64)
#             self.gin_conv_128 = GINConv(self.mesh_mlp_128)

#     def forward(self, nodes_features, edge_index):
#         x = nodes_features
#         edge_index = edge_index
#         x1 = self.gin_conv_256(x, edge_index)  # 64 256 1024
#         # x1 = x1.view(x1.size(0), -1)
#         # x1 = torch.max(x1, dim=2)[0]
#         return self.classifier(x1)


# class MeshGraph(nn.Module):
#     """Some Information about MeshGraph"""

#     def __init__(self, opt):
#         super(MeshGraph, self).__init__()
#         self.spatial_descriptor = SpatialDescriptor()
#         self.structural_descriptor = StructuralDescriptor()
#         self.mesh_conv1 = MeshConvolution(64, 131, 256, 256)
#         self.mesh_conv2 = MeshConvolution(256, 256, 512, 512)
#         self.fusion_mlp = nn.Sequential(
#             nn.Conv1d(1024, 1024, 1),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#         )
#         self.concat_mlp = nn.Sequential(
#             nn.Conv1d(1792, 1024, 1),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(256, 40)
#         )

#     def forward(self, nodes_features, edge_index, centers, corners, normals, neigbour_index):
#         spatial_fea0 = self.spatial_descriptor(centers)
#         structural_fea0 = self.structural_descriptor(
#             corners, normals)

#         spatial_fea1, structural_fea1 = self.mesh_conv1(
#             spatial_fea0, structural_fea0)
#         spatial_fea2, structural_fea2 = self.mesh_conv2(
#             spatial_fea1, structural_fea1)

#         spatial_fea3 = self.fusion_mlp(
#             torch.cat([spatial_fea2, structural_fea2], 1))

#         fea = self.concat_mlp(
#             torch.cat([spatial_fea1, spatial_fea2, spatial_fea3], 1))

#         fea = torch.max(fea, dim=2)[0]
#         fea = fea.reshape(fea.size(0), -1)
#         fea = self.classifier[:-1](fea)
#         cls = self.classifier[-1:](fea)
#         return cls


class MeshGraph(nn.Module):
    """Some Information about MeshGraph"""

    def __init__(self, opt):
        super(MeshGraph, self).__init__()
        self.opt = opt
        self.spatial_descriptor = SpatialDescriptor()
        self.structural_descriptor = StructuralDescriptor()
        self.shape_descriptor = NormGraphConv()

        self.mesh_conv1 = MeshConvolution(64, 131, 256, 256)
        self.mesh_conv2 = MeshConvolution(256, 256, 512, 512)
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(1792, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 40)
        )

    def forward(self, nodes_features, edge_index, centers, corners, normals, neigbour_index):
        shape_norm = self.shape_descriptor(
            nodes_features, edge_index, self.opt)  # n 64 1024

        spatial_fea0 = self.spatial_descriptor(centers)
        structural_fea0 = self.structural_descriptor(
            corners, normals)

        spatial_fea1, structural_fea1 = self.mesh_conv1(
            spatial_fea0, structural_fea0, shape_norm)
        spatial_fea2, structural_fea2 = self.mesh_conv2(
            spatial_fea1, structural_fea1, shape_norm)

        spatial_fea3 = self.fusion_mlp(
            torch.cat([spatial_fea2, structural_fea2], 1))

        fea = self.concat_mlp(
            torch.cat([spatial_fea1, spatial_fea2, spatial_fea3], 1))

        fea = torch.max(fea, dim=2)[0]
        fea = fea.reshape(fea.size(0), -1)
        fea = self.classifier[:-1](fea)
        cls = self.classifier[-1:](fea)
        return cls, fea
