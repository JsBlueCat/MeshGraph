import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class FaceRotateConvolution(nn.Module):

    def __init__(self):
        super(FaceRotateConvolution, self).__init__()
        self.rotate_mlp = nn.Sequential(
            nn.Conv1d(6, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, corners):
        fea = (self.rotate_mlp(corners[:, :6]) +
               self.rotate_mlp(corners[:, 3:9]) +
               self.rotate_mlp(torch.cat([corners[:, 6:], corners[:, :3]], 1))) / 3
        return self.fusion_mlp(fea)


class SpatialDescriptor(nn.Module):

    def __init__(self):
        super(SpatialDescriptor, self).__init__()

        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, centers):
        return self.spatial_mlp(centers)


class StructuralDescriptor(nn.Module):

    def __init__(self):
        super(StructuralDescriptor, self).__init__()

        self.FRC = FaceRotateConvolution()
        self.structural_mlp = nn.Sequential(
            nn.Conv1d(64+3+64, 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU(),
            nn.Conv1d(131, 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU(),
        )

    def forward(self, corners, normals, extra_norm):
        structural_fea1 = self.FRC(corners)

        return self.structural_mlp(torch.cat([structural_fea1, normals, extra_norm], 1))


class MeshConvolution(nn.Module):

    def __init__(self, spatial_in_channel, structural_in_channel, spatial_out_channel, structural_out_channel):
        super(MeshConvolution, self).__init__()

        self.spatial_in_channel = spatial_in_channel
        self.structural_in_channel = structural_in_channel
        self.spatial_out_channel = spatial_out_channel
        self.structural_out_channel = structural_out_channel

        self.combination_mlp = nn.Sequential(
            nn.Conv1d(self.spatial_in_channel +
                      self.structural_in_channel, self.spatial_out_channel, 1),
            nn.GroupNorm(32, self.spatial_out_channel),
            nn.ReLU(),
        )

        self.aggregation_mlp = nn.Sequential(
            nn.Conv1d(self.structural_in_channel,
                      self.structural_out_channel, 1),
            nn.GroupNorm(32, self.structural_out_channel),
            nn.ReLU(),
        )

    def forward(self, spatial_fea, structural_fea, _):
        b, _, n = spatial_fea.size()

        # Combination
        spatial_fea = self.combination_mlp(
            torch.cat([spatial_fea, structural_fea], 1))

        # Aggregation
        structural_fea = self.aggregation_mlp(structural_fea)

        return spatial_fea, structural_fea
