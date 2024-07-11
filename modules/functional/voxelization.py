from torch.autograd import Function

from .backend import _backend

__all__ = ['avg_voxelize']


class AvgVoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution):
        """
        Finding the average feature of voxel
        :param ctx:
        :param features: Features of the point cloud, FloatTensor[B, C, N]
        B: batch_size, C: feature_channel, N: point_number in each batch
        :param coords: Voxelized Coordinates of each point, IntTensor[B, 3, N]
        :param resolution: Voxel resolution
        :return:
            Voxelized Features, FloatTensor[B, C, R, R, R]
        """
        features = features.contiguous()
        coords = coords.int().contiguous()
        b, c, _ = features.shape#[b, c, _] -> [B, C, N]
        #out : outputs, FloatTensor[b, c, s], s = r ** 3
        #indices : voxel index of each point, IntTensor[b, n]
        #cnt : #points in each voxel index, IntTensor[b, s]
        out, indices, counts = _backend.avg_voxelize_forward(features, coords, resolution)
        ctx.save_for_backward(indices, counts)#equals to "ctx.indices = indices" and "ctx.counts = counts"
        return out.view(b, c, resolution, resolution, resolution)#return [B, C, R, R, R]

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of output, FloatTensor[B, C, R, R, R]
        :return:
            gradient of inputs, FloatTensor[B, C, N]
        """
        b, c = grad_output.shape[:2]#get [B, C]
        indices, counts = ctx.saved_tensors
        grad_features = _backend.avg_voxelize_backward(grad_output.contiguous().view(b, c, -1), indices, counts)
        return grad_features, None, None


avg_voxelize = AvgVoxelization.apply
