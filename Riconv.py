import torch
from modules.voxelization import Voxelization
from modules.devoxelization import DeVoxelization
from modules.riconv2_utils import RIConv2SetAbstraction, compute_LRA, RIConv2FeaturePropagation_v2
from torch import nn
import torch.nn.functional as F


class RIconv2SetAbstraction_Unter(nn.Module):
    def __init__(self,npoint = [128, 64, 32, 16], radius = [0.2, 0.4, 0.6, 0.9], nsample = [4, 8, 16, 32], in_channel = [0 + 64, 64 + 64, 128 + 64, 256 + 64], mlp = [[32], [64], [128], [256]], group_all=False):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.normal_channel = True
        self.depth = 4
        self.resolution = [32, 16, 8, 4]
        self.prev_mlp_convs = nn.ModuleList()
        self.prev_mlp_bns = nn.ModuleList()
        
        self.sa0 = RIConv2SetAbstraction(npoint=npoint[0], radius=radius[0],  nsample=nsample[0], in_channel=in_channel[0], mlp=[64],  group_all=False)
        self.sa1 = RIConv2SetAbstraction(npoint=npoint[1],  radius=radius[1], nsample=nsample[1], in_channel=in_channel[1], mlp=[128],  group_all=False)
        self.sa2 = RIConv2SetAbstraction(npoint=npoint[2],  radius=radius[2], nsample=nsample[2], in_channel=in_channel[2], mlp=[256],  group_all=False)
        self.sa3 = RIConv2SetAbstraction(npoint=npoint[3],  radius=radius[3],  nsample=nsample[3], in_channel=in_channel[3], mlp=[512],  group_all=False)
        
        self.fp3 = RIConv2FeaturePropagation_v2(radius=1.5, nsample=4, in_channel=512+64, in_channel_2=512+256, mlp=[512], mlp2=[256])
        self.fp2 = RIConv2FeaturePropagation_v2(radius=0.8, nsample=8, in_channel=256+64, in_channel_2=256+128, mlp=[256], mlp2=[128])
        self.fp1 = RIConv2FeaturePropagation_v2(radius=0.5, nsample=16, in_channel=128+64, in_channel_2=128+64, mlp=[128], mlp2=[64])
        self.fp0 = RIConv2FeaturePropagation_v2(radius=0.5, nsample=32,  in_channel=64+64, in_channel_2=64, mlp=[64], mlp2=[])

        self.conv1 = nn.Conv1d(64, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv1d(32, 1, 1)
        self.Sigmoid = nn.Sigmoid()


    def forward(self, xyz, feature):
        VPoints = []
        norm = xyz[:, :, 3:]
        xyz = xyz[:, :, :3]
        feature = feature
        
        l0_xyz, l0_norm, l0_points, feature = self.sa0(xyz, norm, None, feature)
        VPoints.append(self.Voxelization(l0_xyz, l0_points, self.resolution[0])[0])
        l1_xyz, l1_norm, l1_points, feature = self.sa1(l0_xyz, l0_norm, l0_points, feature)
        VPoints.append(self.Voxelization(l1_xyz, l1_points, self.resolution[1])[0])
        l2_xyz, l2_norm, l2_points, feature = self.sa2(l1_xyz, l1_norm, l1_points, feature)
        VPoints.append(self.Voxelization(l2_xyz, l2_points, self.resolution[2])[0])
        l3_xyz, l3_norm, l3_points, feature = self.sa3(l2_xyz, l2_norm, l2_points, feature)
        VPoints.append(self.Voxelization(l3_xyz, l3_points, self.resolution[3])[0])


        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_norm, l3_norm, l2_points, l3_points)
        VPoints.append(self.Voxelization(l2_xyz, l2_points, self.resolution[2])[0])
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_norm, l2_norm, l1_points, l2_points)
        VPoints.append(self.Voxelization(l1_xyz, l1_points, self.resolution[1])[0])
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_norm, l1_norm, l0_points, l1_points)
        VPoints.append(self.Voxelization(l0_xyz, l0_points, self.resolution[0])[0])
        point = self.fp0(xyz, l0_xyz, norm, l0_norm, None, l0_points)

        feat =  F.relu(self.bn1(self.conv1(point)))
        x = self.drop1(feat)
        x = self.conv2(x)
        #x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        x = self.Sigmoid(x)
        return x,VPoints

    def Voxelization(self, xyz, point,resolution):
        Vox = Voxelization(resolution=resolution)
        return Vox(point,xyz.transpose(-2,-1))

    def DeVoxelization(self, xyz, Feature, resolution):
        Point = DeVoxelization(resolution=resolution)
        return Point(Feature,xyz.transpose(-2,-1))


if __name__ == '__main__':
    RI_test = torch.rand(8, 512, 3).cuda()
    norm = compute_LRA(RI_test, True)
    RI_test = torch.cat([RI_test, norm], dim=-1)
    RI_Ab = RIconv2SetAbstraction_Unter().cuda()
    result = RI_Ab(RI_test)
    for points in result:
        print(points.shape)