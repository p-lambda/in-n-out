import torch
import torch.nn as nn


class ConvConv(nn.Module):
    '''
    Convenience class for stacking two convoluational layers together, the
    second of which has identical numbers of input and output channels.
    '''
    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        '''
        Constructs a pair of (convolution, BatchNorm, ReLU) triplets.

        Parameters
        ----------
        in_channels : int
            Number of input channels for the first layer.
        out_channels : int
            Number of output channels for the first layer. This also serves as
            the number of input and output channels for the second layer.
        bn_momentum : float
            Batch norm momentum.
        '''
        super(ConvConv, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.conv_conv(X)


class DownConv(nn.Module):
    '''
    Performs (convolution => BatchNorm => ReLU) * 2 => MaxPool.
    '''
    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv_conv = ConvConv(in_channels, out_channels, bn_momentum)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        conv_output = self.conv_conv(X)
        return self.max_pool(conv_output), conv_output


class UpConvConcat(nn.Module):
    '''
    Performs a transposed convolution, concatenates the output of a previous
    convolutional layer, and then applies (convolution => ReLU) * 2 to the
    concatenated data.
    '''

    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        '''
        Constructor.

        Parameters
        ----------
        in_channels : int
            The number of input channels to the convolutional layers (including
            the transposed convolution).
        out_channels : int
            The number of output channels from the convolutional layers.
        bn_momentum : float, default 0.1
            Momentum for the batch norm layers.
        '''
        super(UpConvConcat, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=2, stride=2)
        self.conv = ConvConv(in_channels, out_channels, bn_momentum)

    def forward(self, X, prev_output):
        '''
        Parameters
        ----------
        X : torch.Tensor
            Input to the transposed convoluational layer.
        prev_output : torch.Tensor
            Output from a previous convoluational layer to concatenate.

        Returns
        -------
        torch.Tensor
        '''
        X = self.upconv(X)  # Transposed convolution.
        X_dim = X.size()[-2]
        cropped_output = extract_img(X_dim, prev_output)  # Crop image.
        cat_input = torch.cat((X, cropped_output), dim=1)  # Concatenate.
        return self.conv(cat_input)  # Apply (convolution => ReLU) * 2.


def extract_img(size, in_tensor):
    '''
    Crops the exterior of an image. Used during the concatenation phase of the
    UNet architecture so that the two Tensors are of equal height and width.

    Parameters
    ----------
    size : int
        Desired height and width.
    in_tensor : torch.Tensor
        Tensor to crop.

    Returns
    -------
    torch.Tensor
        Cropped Tensor.
    '''
    dim1, dim2 = in_tensor.size()[-2:]
    dim1_start, dim1_end = int((dim1 - size) / 2), int((dim1 + size) / 2)
    dim2_start, dim2_end = int((dim2 - size) / 2), int((dim2 + size) / 2)
    return in_tensor[..., dim1_start:dim1_end, dim2_start:dim2_end]


class UNetFeatureExtractor(nn.Module):
    '''
    Implements the hidden part (i.e., everything but the output layer) of the
    UNet architecture from "U-Net: Convolutional Networks for Biomedical Image
    Segmentation" by Ronneberger et al. The code for this implementation is
    based off that of "Weakly Supervised Deep Learning for Segmentation of
    Remote Sensing Imagery" from Wang et al.
    '''
    def __init__(self, in_channels, out_channels, task, filters=32,
                 bn_momentum=0.1):
        '''
        Constructs a UNet that can perform image prediction or binary
        classification on image input.

        Parameters
        ----------
        in_channels : int
            The number of channels in the input data.
        out_channels : int
            The number of channels in the output data for image prediction.
        filters : int, default 32
            The number of filters in the first convolutional layers.
        bn_momentum : float, default 0.1
            Batch Norm momentum.
        '''
        super(UNetFeatureExtractor, self).__init__()

        # UNet architecture.
        self.conv1 = DownConv(in_channels, filters, bn_momentum)
        self.conv2 = DownConv(filters, filters * 2, bn_momentum)
        self.conv3 = DownConv(filters * 2, filters * 4, bn_momentum)
        self.conv4 = ConvConv(filters * 4, filters * 8, bn_momentum)
        self.upconv1 = UpConvConcat(filters * 8, filters * 4, bn_momentum)
        self.upconv2 = UpConvConcat(filters * 4, filters * 2, bn_momentum)
        self.upconv3 = UpConvConcat(filters * 2, filters, bn_momentum)

        # Predictor heads.
        '''self.image_predictor = nn.Sequential(nn.Conv2d(filters, out_channels,
                                                       kernel_size=1, stride=1,
                                                       padding=0),
                                             nn.Tanh())
        self.binary_predictor = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                              nn.Flatten(),
                                              nn.Linear(filters, 1))'''

    def forward(self, X):
        X, conv1 = self.conv1(X)
        X, conv2 = self.conv2(X)
        X, conv3 = self.conv3(X)
        X = self.conv4(X)
        X = self.upconv1(X, conv3)
        X = self.upconv2(X, conv2)
        X = self.upconv3(X, conv1)
        return X


def make_image_predictor(filters, out_channels):
    conv = nn.Conv2d(filters, out_channels, kernel_size=1, stride=1, padding=0)
    return nn.Sequential(conv, nn.Tanh())


def make_binary_predictor(filters):
    pool = nn.AdaptiveAvgPool2d(1)
    linear = nn.Linear(filters, 1)
    return nn.Sequential(pool, nn.Flatten(), linear)


class UNet(nn.Module):
    '''
    A UNet that can perform image prediction or binary classification on image
    input.
    '''
    def __init__(self, in_channels, out_channels, task, filters=32,
                 bn_momentum=0.1, dropout_prob=0):
        '''
        Constructor.

        Parameters
        ----------
        in_channels : int
            The number of channels in the input data.
        out_channels : int
            The number of channels in the output data for image prediction.
        task : str
            Specifies the type of output from the model (image or binary).
        filters : int, default 32
            The number of filters in the first convolutional layers.
        bn_momentum : float, default 0.1
            Batch Norm momentum.
        dropout_prob : float, default 0
            Dropout probability before task heads.
        '''
        super(UNet, self).__init__()
        if task not in ('image', 'binary'):
            raise ValueError('Invalid "task" parameter')
        self.task = task

        self.extractor = UNetFeatureExtractor(in_channels, filters,
                                              bn_momentum)

        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)
        # Predictor heads.
        self.image_predictor = make_image_predictor(filters, out_channels)
        self.binary_predictor = make_binary_predictor(filters)

    def forward(self, X):
        X = self.extractor(X)

        if self.dropout_prob > 0.0:
            X = self.dropout(X)

        if self.task == 'image':
            return self.image_predictor(X)
        elif self.task == 'binary':
            return self.binary_predictor(X)
        else:
            raise RuntimeError(f'Invalid "task": {self.task}')


class MultitaskUNet(nn.Module):
    def __init__(self, in_channels, out_channel_list, task_list, use_idx=-1,
                 filters=32, bn_momentum=0.1, dropout_prob=0):
        super(MultitaskUNet, self).__init__()
        self.shared_layers = UNetFeatureExtractor(in_channels, filters,
                                                  bn_momentum, dropout_prob)

        self.task_layers = nn.ModuleList()
        for i, task in enumerate(task_list):
            if task_list[i] == 'image':
                head = make_image_predictor(filters, out_channel_list[i])
            elif task_list[i] == 'binary':
                head = make_binary_predictor(filters, dropout_prob)
            else:
                raise ValueError(f'Invalid "task": {task_list[i]}')

            self.task_layers.append(head)

        assert use_idx < len(self.task_layers)
        self.use_idx = use_idx

    def forward(self, x):
        shared_output = self.shared_layers(x)
        if self.use_idx >= 0:
            return self.task_layers[self.use_idx](shared_output)
        elif self.training:
            return [layer(shared_output) for layer in self.task_layers]
        else:
            return self.task_layers[0](shared_output)
