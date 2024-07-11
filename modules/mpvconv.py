import torch.nn as nn

import modules.functional as F
from modules.voxelization import Voxelization
from modules.shared_mlp import SharedMLP
from modules.se import SE3d
# import modules.functional as F
# from voxelization import Voxelization
# from shared_mlp import SharedMLP
# from se import SE3d

__all__ = ['MPVConv']

# MPVConv extends from nn.Module
class MPVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)

        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        voxel_layers2 = [
            nn.Conv3d(out_channels, out_channels, kernel_size // 2, stride=1, padding=kernel_size // 4),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
        ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
            voxel_layers2.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.voxel_layers2 = nn.Sequential(*voxel_layers2)
        # compute point features by MLP
        self.point_features = SharedMLP(in_channels, out_channels)
        self.point_features2 = SharedMLP(out_channels, out_channels)

    def forward(self, inputs):
        #---Initializing Voxel-Point Neuron
        features, coords = inputs
        # voxelize
        voxel_features, voxel_coords = self.voxelization(features, coords)
        # convolve
        voxel_features = self.voxel_layers(voxel_features)
        # devoxelize by trilinear
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        #---Transmission Voxel-Point Neuron
        fused_features1 = voxel_features + self.point_features(features)
        voxel_features2, voxel_coords2 = self.voxelization(fused_features1, coords)
        voxel_features2 = self.voxel_layers2(voxel_features2)
        voxel_features2 = F.trilinear_devoxelize(voxel_features2, voxel_coords2, self.resolution, self.training)
        fused_features = voxel_features2 + self.point_features2(fused_features1) + voxel_features
        return fused_features, coords
