"""Reference: Yunyang Xiong, Hyunwoo J. Kim, Varsha Hedau, "ANTNets: Mobile
Convolutional Neural Networks for Resource Efficient Image Classification".
"""

import torch.nn as _nn


class AntBlock(_nn.Module):
    """AntBlock for ANTNet.

    Args:
        inplanes (int): Number of input channels of the feature map.
        outplanes (int): Number of output channels of the feature map.
        expansion (int): Factor for channel expansion in the expansion layer.
        reduction_ratio (int): Factor for channel reduction in the attention
            mask.
        stride (int): Stride for reducing the image resolution in the
            depthwise layer.
        group (int): Number of groups for the Group convolution layer.
    """

    def __init__(self,
                 inplanes,
                 outplanes,
                 expansion,
                 reduction_ratio,
                 stride,
                 group):
        super(AntBlock, self).__init__()
        assert stride in [1, 2], "Stride must be either 1 or 2."
        self.use_res_connect = stride == 1 and inplanes == outplanes
        expplanes = expansion * inplanes
        self.expansion = _nn.Sequential(
            _nn.Conv2d(inplanes, expplanes, 1),
            _nn.ReLU6(inplace=True)
        )

        self.depthwise = _nn.Sequential(
            _nn.Conv2d(expplanes,
                       expplanes,
                       3,
                       stride,
                       1,
                       groups=expplanes),
            _nn.ReLU6(inplace=True)
        )

        self.attention = _nn.Sequential(
            _nn.AdaptiveAvgPool2d((1, 1)),
            _nn.Conv2d(expplanes, expplanes // reduction_ratio, 1),
            _nn.ReLU(inplace=True),
            _nn.Conv2d(expplanes // reduction_ratio, expplanes, 1),
            _nn.Sigmoid()
        )

        self.projection = _nn.Conv2d(expplanes, outplanes, 1, groups=group)

    def forward(self, x):
        identity = x
        out = self.expansion(x)
        out = self.depthwise(out)
        out1 = self.attention(out)
        out2 = out * out1
        out2 = self.projection(out2)
        if self.use_res_connect:
            out2 += identity
        return out2


class AntNet(_nn.Module):
    """ANTNet architecture.

    Args:
        in_channels (int): Number of input channels in the image.
        num_classes (int): Number of classes.
        num_stages (int): Number of Ant Block stages.
        outplanes (list): Number of output channels across conv layers and Ant
            block stages.
        repetitions (list): Number of blocks for all Ant block stages.
        expansions (list): Channel expansion factors for Ant block stages.
        reduction_ratios (list): Channel reduction ratios for Ant block stages.
        strides (list): Strides for conv layers and Ant block stages.
        groups (list): Number of groups for Ant block stages.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 num_stages,
                 outplanes,
                 repetitions,
                 expansions,
                 reduction_ratios,
                 strides,
                 groups):
        super(AntNet, self).__init__()
        self.num_stages = num_stages
        assert self.num_stages == len(repetitions) == len(expansions) == \
            len(reduction_ratios) == len(groups), "Configuration error."
        self.conv0 = _nn.Conv2d(in_channels,
                                outplanes[0],
                                3,
                                strides[0],
                                1)
        self.add_module('conv0', self.conv0)
        # Ant Blocks
        self.ant_names = list()
        for i in range(self.num_stages):
            num_blocks = repetitions[i]
            for j in range(num_blocks):
                # First block of a stage changes the dimensions of image
                inplanes = outplanes[i] if j == 0 else outplanes[i + 1]
                stride = strides[i + 1] if j == 0 else 1
                ant_block = AntBlock(inplanes,
                                     outplanes[i + 1],
                                     expansions[i],
                                     reduction_ratios[i],
                                     stride,
                                     groups[i])
                block_name = 'ant{}_{}'.format(i + 1, j + 1)
                self.add_module(block_name, ant_block)
                self.ant_names.append(block_name)
        self.conv8 = _nn.Conv2d(outplanes[i + 1],
                                outplanes[i + 2],
                                1,
                                strides[i + 2])
        self.add_module('conv8', self.conv8)
        self.pool9 = _nn.AdaptiveAvgPool2d((1, 1))
        self.fc10 = _nn.Linear(outplanes[i + 2], num_classes)

    def forward(self, inp):
        x = self.conv0(inp)
        for i, block_name in enumerate(self.ant_names):
            block = getattr(self, block_name)
            x = block(x)
        x = self.conv8(x)
        x = self.pool9(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc10(x)
        return x
