from torch import nn
import torch
from typing import Tuple, Union
from modules.neural_network import SegmentationNetwork
from modules.dynunet_block import UnetOutBlock, UnetResBlock
from modules.model_components import DRIVPocketEncoder, DRIVPocketUPBLock, Last_DRIVPocketUPBLock
import numpy as np
from Riconv import RIconv2SetAbstraction_Unter

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
                                   nn.GroupNorm(1,out_channels),
                                   nn.ReLU(),
                                   nn.Dropout3d(0.1),
                                   nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
                                   nn.GroupNorm(1,out_channels),
                                   nn.ReLU())

    def forward(self, x):
        out = self.block(x)
        return out
class DRIVPocket(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            feature_size: int = 64,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "batch",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=False,
            Riconv_att=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimensions of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        """
        super().__init__()
        self.RIconv2_Unter = None
        if Riconv_att:
            self.RIconv2_Unter = RIconv2SetAbstraction_Unter()
        if depths is None:
            depths = [1, 1, 1, 1]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.feat_size = (4, 4, 4,)
        self.hidden_size = hidden_size
        #self.in1 = DoubleConv(14, 32, 3)

        self.DRIVPocket_encoder = DRIVPocketEncoder(in_channels=feature_size,dims=dims, depths=depths, num_heads=num_heads,Riconv_att=Riconv_att)
        #self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads,Riconv_att=False)
        self.sigmoid = nn.Sigmoid()

        self.encoder1 = DoubleConv(
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
        )
        self.decoder5 = DRIVPocketUPBLock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            depth=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8*8*8,
            Riconv_att=Riconv_att,
            padding=1,
            outpad=1
        )
        self.decoder4 = DRIVPocketUPBLock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            depth=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16*16*16,
            Riconv_att=Riconv_att,
            padding=1,
            outpad=1,
        )
        self.decoder3 = DRIVPocketUPBLock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            depth=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32*32*32,
            Riconv_att=Riconv_att,
            padding=1,
            outpad=1,
        )
        self.decoder2 = Last_DRIVPocketUPBLock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            depth=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=65*65*65,
            Riconv_att=Riconv_att,
            conv_decoder=True
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in, point_in, feature):
        #print("###########reached forward network")
        #print("XIN",x_in.shape)

        convBlock = self.encoder1(x_in)

        if self.RIconv2_Unter is not None:
            point_output, RiconVox = self.RIconv2_Unter(point_in, feature)
            x_output, hidden_states = self.DRIVPocket_encoder(convBlock, RiconVox)
        else:
            x_output, hidden_states = self.DRIVPocket_encoder(convBlock)

        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]
        dec4 = enc4
        if self.RIconv2_Unter is not None:
            dec3 = self.decoder5(dec4, enc3, RiconVox[4])
            dec2 = self.decoder4(dec3, enc2, RiconVox[5])
            dec1 = self.decoder3(dec2, enc1, RiconVox[6])
        else:
            dec3 = self.decoder5(dec4, enc3)
            dec2 = self.decoder4(dec3, enc2)
            dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1,convBlock)

        logits = self.out1(out)
        if self.RIconv2_Unter is not None:
            return point_output,self.sigmoid(logits)
        else:
            return self.sigmoid(logits)
