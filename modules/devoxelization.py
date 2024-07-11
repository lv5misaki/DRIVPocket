import torch
import torch.nn as nn

# import modules.functional as F
import modules.functional as F

__all__ = ['DeVoxelization']


class DeVoxelization(nn.Module):
    def __init__(self, resolution, normalize=False, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        coords = coords.detach()
        # The coords are normalized to the local coordinate system, the mean value is subtracted first, and the dimension 32 * 22 * 2048 is input（batch_size * coord_feature * num_point）
        norm_coords = coords - coords.mean(2, keepdim=True)
        #features = features.detach()
        #norm_features = features - features.mean(2, keepdim=True)

        if self.normalize:
            # Find the farthest point as the radius, then divide each point by 2 * radius, normalize the coordinates to [- 0.5,0.5], add 0.5, and convert it to [0.0,1.0]
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
            #norm_features = norm_features / (norm_features.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        # resolution is positive integer，clamp norm_coords form [0,1] to [0,r-1]
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        # Through round, we get vox_coords which is an integer of [0, R-1] with a total of R values. The default value of r is 32
        vox_coords = torch.round(norm_coords).to(torch.float32)
        # forward computing for voxelization
        return F.trilinear_devoxelize(features, vox_coords, self.r)

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')
