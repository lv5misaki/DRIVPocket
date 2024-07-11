import torch
from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.utils import optional_import
from transformerblock import TransformerBlock
from dynunet_block import get_conv_layer, UnetResBlock, UnetUpBlock


einops, _ = optional_import("einops")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
                                   nn.GroupNorm(1,out_channels),
                                   nn.ReLU(),
                                   nn.Dropout3d(0.1),
                                   nn.Conv3d(out_channels, out_channels, 1, padding=0),
                                   nn.GroupNorm(1,out_channels),
                                   nn.ReLU())

    def forward(self, x):
        out = self.block(x)
        return out
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride=2,padding = 1):
        super().__init__()
        self.block = nn.Sequential(nn.Sequential(nn.MaxPool3d(kernel_size, stride=stride,padding=padding), DoubleConv(in_channels, out_channels, 3)))

    def forward(self, x):
        out = self.block(x)
        return out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_up,padding=0,stride=2, out_pad=0, upsample=None):
        super().__init__()
        if upsample:
            self.up_s = nn.Upsample(scale_factor=2, mode=upsample, align_corners=True)
        else:
            self.up_s = nn.ConvTranspose3d(in_channels, out_channels, 3, stride=2, padding=padding,
                                           output_padding=out_pad)

        self.convT = DoubleConv(in_channels, out_channels, 3)

    def forward(self, x1, x2):
        out = self.up_s(x1)
        out = self.convT(torch.cat((x2, out), dim=1))
        return out

class DRIVPocketEncoder(nn.Module):
    def __init__(self, input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],dims=[32, 64, 128, 256],#这部分目前是in_channels的两倍
                 proj_size =[256,256,128,64], depths=[2, 2, 2, 2],  num_heads=4, spatial_dims=3, in_channels=14,
                 dropout=0.2, transformer_dropout_rate=0.1 , Riconv_att=True,**kwargs):#这的inputsize是下采样后的Transfomer的input
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        #stem_layer = UnetResBlock(spatial_dims, in_channels, dims[0], kernel_size=3, stride=2, norm_name="batch",dropout=dropout)
        stem_layer = Down(32,dims[0],3,2,0)
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = Down(dims[i], dims[i+1],3,2,1)
            self.downsample_layers.append(downsample_layer)
        self.stages = []

        self.stages1 = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple Transformer blocks
        self.stages2 = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple Transformer blocks
        self.stages3 = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple Transformer blocks
        self.stages4 = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for j in range(depths[0]):
            self.stages1.append(TransformerBlock(input_size=input_size[0], hidden_size=dims[0],
                                                proj_size=proj_size[0], num_heads=num_heads,
                                                dropout_rate=transformer_dropout_rate, pos_embed=False, Riconv_att=Riconv_att))
        for j in range(depths[1]):
            self.stages2.append(TransformerBlock(input_size=input_size[1], hidden_size=dims[1],
                                                proj_size=proj_size[2], num_heads=num_heads,
                                                dropout_rate=transformer_dropout_rate, pos_embed=False, Riconv_att=Riconv_att))
        for j in range(depths[2]):
            self.stages3.append(TransformerBlock(input_size=input_size[2], hidden_size=dims[2],
                                                proj_size=proj_size[2], num_heads=num_heads,
                                                dropout_rate=transformer_dropout_rate, pos_embed=False, Riconv_att=Riconv_att))
        for j in range(depths[3]):
            self.stages4.append(TransformerBlock(input_size=input_size[3], hidden_size=dims[3],
                                                proj_size=proj_size[3], num_heads=num_heads,
                                                dropout_rate=transformer_dropout_rate, pos_embed=False, Riconv_att=Riconv_att))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x, riconv=None):
        hidden_states = []#用于shortcut的部分

        x = self.downsample_layers[0](x)
        for stage in self.stages1:
            if riconv is not None:
                x = stage(torch.stack([x, riconv[0]], dim = 0))
            else:
                x = stage(x)

        hidden_states.append(x)

        x = self.downsample_layers[1](x)
        for stage in self.stages2:
            if riconv is not None:
                x = stage(torch.stack([x, riconv[1]], dim = 0))
            else:
                x = stage(x)
        hidden_states.append(x)

        x = self.downsample_layers[2](x)
        for stage in self.stages3:
            if riconv is not None:
                x = stage(torch.stack([x, riconv[2]], dim = 0))
            else:
                x = stage(x)
        hidden_states.append(x)

        x = self.downsample_layers[3](x)
        for stage in self.stages4:
            if riconv is not None:
                x = stage(torch.stack([x, riconv[3]], dim = 0))
            else:
                x = stage(x)
        hidden_states.append(x)

        return x, hidden_states

    def forward(self, x, Riconv=None):
        x, hidden_states = self.forward_features(x, Riconv)
        return x, hidden_states


class DRIVPocketUPBLock(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 2,
            conv_decoder: bool = False,
            Riconv_att = False,
            padding = 0,
            outpad=0,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        # self.transp_conv = get_conv_layer(
        #     spatial_dims,
        #     in_channels,
        #     out_channels,
        #     kernel_size=upsample_kernel_size,
        #     stride=upsample_stride,
        #     conv_only=True,
        #     is_transposed=True,
        # )
        self.transp_conv = Up(
            in_channels,
            out_channels,
            3,
            padding=padding,
            out_pad=outpad,
        )
        self.depth = depth
        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_blocks = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_blocks.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                self.decoder_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels,
                                                     proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=False, Riconv_att = Riconv_att))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inp, skip, riconv=None):

        out = self.transp_conv(inp,skip)
        if riconv is not None:
            for decoder_block in self.decoder_blocks:
                out = decoder_block(torch.stack([out,riconv], dim = 0))
        else:
            for decoder_block in self.decoder_blocks:
                out = decoder_block(out)

        return out
class Last_DRIVPocketUPBLock(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 1,
            conv_decoder: bool = False,
            Riconv_att=False
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = nn.ConvTranspose3d(in_channels, out_channels, 3, stride=2, padding=0,
                                           output_padding=0)
        self.convT = DoubleConv(in_channels, out_channels, 3)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_blocks = nn.ModuleList()
        self.depth = depth
        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_blocks.append(
                UnetResBlock(3, out_channels, out_channels, kernel_size=3, stride=1, norm_name=("group", {"num_groups": 1})))
        else:
            stage_blocks = []
            for j in range(depth):
                self.decoder_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels,
                                                     proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True, Riconv_att = Riconv_att))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, inp, skip, riconv=None):

        out = self.transp_conv(inp)
        out = self.convT(torch.cat((skip, out), dim=1))
        if riconv is not None:
            for decoder_block in self.decoder_blocks:
                out = decoder_block(torch.stack([out,riconv], dim = 0))
        else:
            for decoder_block in self.decoder_blocks:
                out = decoder_block(out)

        return out