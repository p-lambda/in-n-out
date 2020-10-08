from collections import OrderedDict

from torch import nn

from innout.models import MultitaskModel


class CNN1DFeatureExtractor(nn.Module):
    def __init__(self, in_channels, output_size=128, batch_norm=False):
        super().__init__()
        self.output_size = output_size
        self.in_channels = in_channels

        activ = nn.ReLU(True)

        if batch_norm:
            self.feature_extractor = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels, 32, 5, padding=2)),
                ('bn1', nn.BatchNorm1d(32)),
                ('relu1', activ),
                ('conv2', nn.Conv1d(32, 32, 3, padding=1)),
                ('bn2', nn.BatchNorm1d(32)),
                ('relu2', activ),
                ('maxpool1', nn.MaxPool1d(2, 2)),
                ('conv3', nn.Conv1d(32, 64, 3, padding=1)),
                ('bn3', nn.BatchNorm1d(64)),
                ('relu3', activ),
                ('maxpool2', nn.MaxPool1d(2, 2)),
                ('conv4', nn.Conv1d(64, output_size, 3, padding=1)),
                ('bn4', nn.BatchNorm1d(output_size)),
                ('relu4', activ),
                ('avgpool', nn.AdaptiveAvgPool1d(1)),
            ]))
        else:
            self.feature_extractor = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels, 32, 5, padding=2)),
                ('relu1', activ),
                ('conv2', nn.Conv1d(32, 32, 3, padding=1)),
                ('relu2', activ),
                ('maxpool1', nn.MaxPool1d(2, 2)),
                ('conv3', nn.Conv1d(32, 64, 3, padding=1)),
                ('relu3', activ),
                ('maxpool2', nn.MaxPool1d(2, 2)),
                ('conv4', nn.Conv1d(64, output_size, 3, padding=1)),
                ('relu4', activ),
                ('avgpool', nn.AdaptiveAvgPool1d(1)),
            ]))

    def forward(self, x):
        features = self.feature_extractor(x).view(-1, self.output_size)
        return features


class CNN1D(nn.Module):
    '''
    CNN for time series classification
    '''

    def __init__(self, output_size, in_channels, batch_norm=False, dropout_prob=0.0):
        super().__init__()
        self.output_size = output_size
        self.in_channels = in_channels
        self.dropout_prob = dropout_prob

        activ = nn.ReLU(True)

        self.feature_extractor = CNN1DFeatureExtractor(in_channels, output_size=128,
                                                       batch_norm=False)

        if batch_norm:
            self.classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(128, 1024)),
                ('bn1', nn.BatchNorm1D(1024)),
                ('relu1', activ),
                ('fc2', nn.Linear(1024, output_size)),
            ]))
        else:
            self.classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(128, 1024)),
                ('relu1', activ),
                ('fc2', nn.Linear(1024, output_size)),
            ]))

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.dropout_prob > 0.0:
            features = nn.Dropout(self.dropout_prob)(features)
        logits = self.classifier(features)
        return logits


class CNN1DMultitask(MultitaskModel):
    '''
    CNN1D with Multitask heads
    '''
    def __init__(self, in_channels, task_dims, batch_norm=False, use_idx=None, dropout_prob=0.0):
        '''
        Constructor.

        Parameters
        ----------
        shared_dims : List[int]
            Defines the number and sizes of hidden layers that are shared
            amongst the tasks.
        task_dims : List[List[int]]
            Defines the number and sizes of hidden layers for a variable number
            of tasks.
        '''
        feature_size = 128
        shared_layers = CNN1DFeatureExtractor(in_channels, output_size=feature_size, batch_norm=batch_norm)
        super().__init__(feature_size, task_dims, shared_layers, batch_norm=batch_norm, use_idx=use_idx, dropout_prob=dropout_prob)
