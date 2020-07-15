import torch
import torch.nn as nn


class FaceVectorConv(nn.Module):
    def __init__(self, output_channel=64):
        super(FaceVectorConv, self).__init__()
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
            nn.Conv1d(64, output_channel, 1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU()
        )

    def forward(self, x, opt):
        '''
        x : batch_size *15 , 1024
        center ox oy oz norm
        3       3  3  3   3
        '''
        data = x.view(opt.batch_size, -1, 15).transpose(1, 2)
        xy = data[:, 3:9]
        yz = data[:, 6:12]
        xz = torch.cat([data[:, 3:6], data[:, 9:12]], dim=1)
        face_line = (
            self.rotate_mlp(xy) +
            self.rotate_mlp(yz) +
            self.rotate_mlp(xz)
        ) / 3  # 64 , 64 , 1024
        return self.fusion_mlp(face_line)


class PointConv(nn.Module):
    def __init__(self):
        super(PointConv, self).__init__()

        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, x, opt):
        '''
        center ox oy oz norm
        '''
        data = x.view(opt.batch_size, -1, 15).transpose(1, 2)
        x = data[:, :3]
        return self.spatial_mlp(x)


class MeshMlp(nn.Module):
    def __init__(self, opt, output_channel=256):
        super(MeshMlp, self).__init__()
        self.opt = opt
        self.pc = PointConv()
        self.fvc = FaceVectorConv()
        self.mlp = nn.Sequential(
            nn.Conv1d(131, output_channel, 1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            nn.Conv1d(output_channel, output_channel, 1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU()
        )

    def forward(self, x):
        # x: batch_size*15,1024
        data = x.view(self.opt.batch_size, -1, 15).transpose(1, 2)
        point_feature = self.pc(x, self.opt)  # n 64 1024
        face_feature = self.fvc(x, self.opt)  # n 64 1024
        norm = data[:, 12:]
        fusion_feature = torch.cat(
            [norm, point_feature, face_feature]  # n 64+64+3 = 131 1024
            , dim=1
        )
        return self.mlp(fusion_feature)  # n 256 1024


class NormMlp(nn.Module):
    def __init__(self, opt):
        super(NormMlp, self).__init__()
        self.opt = opt
        self.extra_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, x):
        data = x.view(self.opt.batch_size, -1, 15).transpose(1, 2)
        norm = data[:, 12:]
        return self.extra_mlp(norm)  # n 64 1024
