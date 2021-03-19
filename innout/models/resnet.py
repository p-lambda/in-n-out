import torch
import torch.nn as nn
from innout.models import MLP
from innout.models import MultitaskModel

# from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, num_channels=3, only_feats=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.only_feats = only_feats
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def get_feats(self, x, layer=4):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if layer == 1:
            return x
        x = self.layer2(x)
        if layer == 2:
            return x
        x = self.layer3(x)
        if layer == 3:
            return x
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


    def _forward_impl(self, x, with_feats=False):
        feats = self.get_feats(x)
        x = self.fc(feats)

        if with_feats:
            return x, feats
        else:
            return x

    def forward(self, x, with_feats=False):
        if self.only_feats:
            return self.get_feats(x)
        else:
            return self._forward_impl(x, with_feats)


class ResNet18(ResNet):
    def __init__(self, num_classes=10, num_channels=3, only_feats=False):
        super().__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
            num_channels=num_channels, only_feats=only_feats)

class ResNet34(ResNet):
    def __init__(self, num_classes=10, num_channels=3):
        super().__init__(
            BasicBlock, [3, 4, 6, 3], num_classes=num_classes, num_channels=num_channels)

class ResNet50(ResNet):
    def __init__(self, num_classes=10, num_channels=3):
        super().__init__(
            Bottleneck, [3, 4, 23, 3], num_classes=num_classes, num_channels=num_channels)

class ResNet101(ResNet):
    def __init__(self, num_classes=10, num_channels=3):
        super().__init__(
            Bottleneck, [3, 4, 23, 3], num_classes=num_classes, num_channels=num_channels)

class ResNet152(ResNet):
    def __init__(self, num_classes=10, num_channels=3):
        super().__init__(
            Bottleneck, [3, 8, 36, 3], num_classes=num_classes, num_channels=num_channels)


DEPTH_TO_MODEL = {18: ResNet18, 34: ResNet34, 50: ResNet50, 101: ResNet101, 152: ResNet152}


class ResNetWithAttributes(ResNet):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, num_channels=3, pre_attribute_mlp_hiddens_and_output=None, post_attribute_mlp_hiddens=None,
                 num_attributes=0, layer=4, use_attributes=True, only_train_post_attributes=False):
        super().__init__(
            block, layers, num_classes, zero_init_residual, groups, width_per_group,
            replace_stride_with_dilation, norm_layer, num_channels)
        self._block = block
        self._pre_attribute_mlp = None
        self._layer = layer
        self._use_attributes = use_attributes
        self._only_train_post_attributes = only_train_post_attributes
        if self._layer == 4:
            self._resnet_feature_size = 512 * self._block.expansion
        elif self._layer == 3:
            self._resnet_feature_size = 4096 * self._block.expansion
        else:
            raise NotImplementedError('At the moment layer must be 3 or 4.')
        if pre_attribute_mlp_hiddens_and_output is not None:
            # Feed the ResNet features through an MLP to process the features and reduce dimensionality.
            pre_attribute_dims = ([self._resnet_feature_size] +
                                  pre_attribute_mlp_hiddens_and_output[:-1])
            self._pre_attribute_mlp = MLP(
                dims=pre_attribute_dims, output_dim=pre_attribute_mlp_hiddens_and_output[-1])
            feature_dim = pre_attribute_mlp_hiddens_and_output[-1]
        else:
            feature_dim = self._resnet_feature_size
        if post_attribute_mlp_hiddens is not None:
            # Create two MLPs, one which uses the attributes, and one which does not.
            # If use_attributes is True, we will use the first one, otherwise the second.
            post_attribute_mlp_dims = [feature_dim + num_attributes] + post_attribute_mlp_hiddens
            self._post_attribute_mlp = MLP(
                dims=post_attribute_mlp_dims, output_dim=num_classes)
            post_no_attribute_mlp_dims = [feature_dim] + post_attribute_mlp_hiddens
            self._post_no_attribute_mlp = MLP(
                dims=post_no_attribute_mlp_dims, output_dim=num_classes)
        else:
            # Create two linear layers, one which uses the attributes, and one which does not.
            # If use_attributes is True, we will use the first one, otherwise the second.
            self._post_attribute_mlp = nn.Linear(feature_dim + num_attributes, num_classes)
            self._post_no_attribute_mlp = nn.Linear(feature_dim, num_classes)

    def trainable_params(self):
        if self._only_train_post_attributes:
            return self._post_attribute_mlp.parameters()
        else:
            return self.parameters()

    def forward(self, x_and_a, with_feats=False):
        x, a = x_and_a
        feats = self.get_feats(x, layer=self._layer)
        if self._layer == 3:
            feats = feats.view(feats.shape[0], -1)
        assert(feats.shape[1] == self._resnet_feature_size)
        assert(feats.shape[0] == a.shape[0])
        if self._pre_attribute_mlp is not None:
            feats = self._pre_attribute_mlp.forward(feats)
        if self._use_attributes:
            feats_with_a = torch.cat((feats, a.float()), 1)
            x = self._post_attribute_mlp(feats_with_a)
        else:
            x = self._post_no_attribute_mlp(feats)
        if with_feats:
            return x, feats
        else:
            return x


class ResNetWithAttributes18(ResNetWithAttributes):
    def __init__(self, num_classes=10, num_channels=3, pre_attribute_mlp_hiddens_and_output=None,
                 post_attribute_mlp_hiddens=None, num_attributes=0, layer=4, use_attributes=True,
                 only_train_post_attributes=False):
        super().__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_channels=num_channels,
            pre_attribute_mlp_hiddens_and_output=pre_attribute_mlp_hiddens_and_output,
            post_attribute_mlp_hiddens=post_attribute_mlp_hiddens,
            num_attributes=num_attributes, layer=layer, use_attributes=use_attributes,
            only_train_post_attributes=only_train_post_attributes)


class ResNetTileAttributes(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, num_channels=3, num_attributes=0):
        super().__init__(
            block, layers, num_classes, zero_init_residual, groups, width_per_group,
            replace_stride_with_dilation, norm_layer, num_channels+num_attributes)


    def forward(self, x_and_a, with_feats=False):
        x, a = x_and_a
        d = x.shape
        tiled_a = a.float().unsqueeze(dim=2).unsqueeze(dim=2).repeat(1, 1, d[2], d[3])
        assert(d[0] == tiled_a.shape[0] and len(d) == len(tiled_a.shape))
        model_inputs = torch.cat([x, tiled_a], dim=1)
        return super().forward(model_inputs, with_feats)


class ResNetTileAttributes18(ResNetTileAttributes):
    def __init__(self, num_classes=10, num_channels=3, num_attributes=0):
        super().__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_channels=num_channels,
            num_attributes=num_attributes)


class ResNetMultitask(MultitaskModel):
    '''
    ResNet with Multitask heads
    '''
    def __init__(self, num_channels, task_dims, batch_norm=False,
                 use_idx=None, freeze_shared=False):
        '''
        Constructor.

        Parameters
        ----------
        task_dims : List[List[int]]
            Defines the number and sizes of hidden layers for a variable number
            of tasks.
        '''
        feature_size = 512
        shared_layers = ResNet18(num_channels=num_channels, only_feats=True)
        super().__init__(feature_size, task_dims, shared_layers,
                         batch_norm=batch_norm, use_idx=use_idx,
                         freeze_shared=freeze_shared)
        self._shared_layers = shared_layers

    def get_feats(self, x):
        return self._shared_layers(x)
