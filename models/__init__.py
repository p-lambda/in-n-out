from collections import OrderedDict

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import itertools


class CNN(nn.Module):
    '''
    Generic CNN that can handle different image sizes
    '''

    def __init__(self, output_size, in_channels=3):
        super().__init__()
        self.output_size = output_size
        self.in_channels = in_channels

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 32, 5, padding=2)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3, padding=1)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3, padding=1)),
            ('relu3', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
            ('conv4', nn.Conv2d(64, 128, 3, padding=1)),
            ('relu4', activ),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(128, 1024)),
            ('relu1', activ),
            ('fc2', nn.Linear(1024, output_size)),
        ]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features.view(-1, 128))
        return logits


class MLPDropout(nn.Module):
    '''
    A multilayer perception with ReLU activations and dropout layers.
    '''
    def __init__(self, dims, output_dim, dropout_probs):
        '''
        Constructor.

        Parameters
        ----------
        dims : list[int]
            Specifies the input and hidden layer dimensions.
        output_dim : int
            Specifies the output dimension.
        dropout_probs : list[float]
            Specifies the dropout probability at each layer. The length of this
            list must be equal to the length of dims. If the dropout
            probability of a layer is zero, then the dropout layer is omitted
            altogether.
        '''
        if len(dims) != len(dropout_probs):
            raise ValueError('len(dims) must equal len(dropout_probs)')
        if len(dims) < 1:
            raise ValueError('len(dims) must be at least 1')
        if any(prob < 0 or prob > 1 for prob in dropout_probs):
            raise ValueError('Dropout probabilities must be in [0, 1]')

        super(MLPDropout, self).__init__()
        layers = []
        if dropout_probs[0] > 0:  # Input dropout layer.
            layers.append(('Dropout1', nn.Dropout(p=dropout_probs[0])))

        for i in range(len(dims) - 1):
            layers.append((f'Linear{i + 1}', nn.Linear(dims[i], dims[i + 1])))
            layers.append((f'ReLU{i + 1}', nn.ReLU()))
            if dropout_probs[i + 1] > 0:
                dropout = nn.Dropout(p=dropout_probs[i + 1])
                layers.append((f'Dropout{i + 2}', dropout))

        layers.append((f'Linear{len(dims)}', nn.Linear(dims[-1], output_dim)))
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)


class MLP(MLPDropout):
    '''
    A multilayer perceptron with ReLU activations.
    '''
    def __init__(self, dims, output_dim):
        '''
        Constructor.

        Parameters
        ----------
        dims : List[int]
            Specifies the input and hidden layer dimensions.
        output_dim : int
            Specifies the output dimension.
        '''
        super(MLP, self).__init__(dims, output_dim, [0] * len(dims))


class MultitaskModel(nn.Module):
    '''
    A multilayer perceptron for learning multiple tasks simultaneously.
    The model consists of pairs of fully connected and ReLU layers.
    '''
    def __init__(self, feature_dim, task_dims, shared_layers, batch_norm=False, use_idx=None,
                 freeze_shared=False, dropout_prob=0.0):
        '''
        Constructor.

        Parameters
        ----------
        shared_dims : List[int]
            Defines the number and sizes of hidden layers that are shared
            amongst the tasks.
        task_dims : List[List[int]]
            Defines the number and sizes of hidden layers for a variable number of tasks.
        use_idx: int
            Use only a certain head
        freeze_shared: only make the heads trainable.
        '''
        super().__init__()
        self.use_idx = use_idx
        self.shared_layers = shared_layers
        self.freeze_shared = freeze_shared
        self.task_layers = nn.ModuleList()
        for i in range(len(task_dims)):
            curr_task_layers = []
            linear = nn.Linear(feature_dim, task_dims[i][0])
            curr_task_layers.append((f'Task{i + 1}Linear{1}', linear))
            for j in range(1, len(task_dims[i])):
                if batch_norm:
                    curr_task_layers.append((f'Task{i + 1}BN{j}', nn.BatchNorm1d(task_dims[i][j-1])))
                curr_task_layers.append((f'Task{i + 1}ReLU{j}', nn.ReLU()))
                linear = nn.Linear(task_dims[i][j - 1], task_dims[i][j])
                curr_task_layers.append((f'Task{i + 1}Linear{j + 1}', linear))
            curr_task_sequential = nn.Sequential(OrderedDict(curr_task_layers))
            self.task_layers.append(curr_task_sequential)

        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

    def trainable_params(self):
        if self.freeze_shared:
            return itertools.chain(*[l.parameters() for l in self.task_layers])
        else:
            return self.parameters()

    def forward(self, x):
        if isinstance(x, list) and not self.training:
            x = x[0]

        if isinstance(x, list):
            intermed_outs = [self.shared_layers(xi) for xi in x]
            if self.dropout_prob > 0.0:
                intermed_outs = [self.dropout(out) for out in intermed_outs]
            return [layer(intermed_out) for layer, intermed_out in zip(self.task_layers, intermed_outs)]
        else:
            shared_output = self.shared_layers(x)
            if self.dropout_prob > 0.0:
                shared_output = self.dropout(shared_output)

            if self.use_idx is not None:
                return self.task_layers[self.use_idx](shared_output)

            if self.training:
                return [layer(shared_output) for layer in self.task_layers]
            else:  # For eval, return first task output only.
                return self.task_layers[0](shared_output)


class MLPMultitask(MultitaskModel):
    '''
    A multilayer perceptron for learning multiple tasks simultaneously.
    The model consists of pairs of fully connected and ReLU layers.
    '''
    def __init__(self, shared_dims, task_dims, batch_norm=False, use_idx=None):
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
        shared_layers = []
        for i in range(len(shared_dims) - 1):
            linear = nn.Linear(shared_dims[i], shared_dims[i + 1])
            shared_layers.append((f'SharedLinear{i + 1}', linear))
            if batch_norm:
                shared_layers.append((f'SharedBN{i + 1}', nn.BatchNorm1D(shared_dims[i+1])))
            shared_layers.append((f'SharedReLU{i + 1}', nn.ReLU()))
        linear = nn.Linear(shared_dims[-1], shared_dims[-1])
        shared_layers.append((f'SharedLinearFinal', linear))
        shared_layers = nn.Sequential(OrderedDict(shared_layers))
        super().__init__(shared_dims[-1], task_dims, shared_layers, batch_norm=batch_norm, use_idx=use_idx)


class SyntheticLinearMTL(nn.Module):
    '''
    Simple linear multitask model from "Understanding and Improving Information
    Transfer in Multi-Task Learning." Should probably be merged into
    MLPMultitask at some point.
    '''
    def __init__(self, num_tasks, d, prediction='regression'):
        '''
        Constructor.

        Parameters
        ----------
        num_tasks : int
            The number of output heads for the model, one for each task.
        d : int
            The dimensionality of the input data for the shared linear layer.
        prediction : str, default 'regression'
            Describes the output type of the tasks. Either 'regression',
            'classification', or 'relu'.
        '''
        assert isinstance(num_tasks, int) and num_tasks > 1
        assert isinstance(d, int) and d > 1
        assert prediction in ['regression', 'classification', 'relu']
        self.prediction = prediction
        super(SyntheticLinearMTL, self).__init__()
        self.B = nn.Linear(d, 1)  # Shared linear layer amongst all tasks.
        self.A_list = nn.ModuleList()  # Separate task heads.
        for _ in range(num_tasks):
            self.A_list.append(nn.Linear(1, 1, bias=False))

    def forward(self, x):
        '''
        Takes a batch of data from a single task and computes the output.

        Parameters
        ----------
        x : Dict[str, Union[torch.Tensor, int]
            Has a key 'data' that maps to a batch of PyTorch data and a key
            'task_idx' that specifies to which task this batch belongs.

        Returns
        -------
        out : torch.Tensor
            Output from the shared layer and the specified task layer.
        '''
        data = x['data']
        out = self.B(data)  # All tasks go through shared layer.
        out = self.A_list[x['task_idx']](out)  # Run through given task layer.
        if self.prediction == 'relu':
            out = nn.functional.relu(out)
        return out

#########################
# DEPRECATED
########################
class FCModel(nn.Module):

    def __init__(self, dims):
        """
        Args:
            dims (List[int]): input dim, hidden layer dims, and output dim
        """
        super().__init__()
        if len(dims) < 2:
            raise ValueError("len(dims) must be >= 2")

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, batch):
        out = batch['input'].float()
        for layer in self.layers:
            out = layer(out)
        return out


class ARModel(nn.Module):

    def __init__(self, dims, order):
        super().__init__()
        self.fc = FCModel(dims)
        self.order = order

    def forward(self, batch):
        if not self.training:
            # eval mode only handle batch size 1
            prev_outputs = batch['y0'].float()
            out = prev_outputs.numpy()
            for i in range(batch['input'].shape[1] - self.order):
                curr_out = self.fc.forward({'x': prev_outputs})
                out = np.concatenate([out, curr_out.detach().numpy()], axis=1)
                # prev_outputs = np.r_[prev_outputs.numpy()[1:], curr_out_np]
                prev_outputs = torch.cat([prev_outputs[:, 1:], curr_out], dim=1)
            out = torch.from_numpy(out)
        else:
            # x is the previous true targets in training
            # prev_ys = Variable(batch['input']).float()
            # out = self.fc.forward({'x': prev_ys})

            X = batch['input'].numpy()
            y = batch['target'].numpy()
            theta = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
            self.fc.layers[0].weight = torch.nn.Parameter(torch.from_numpy(theta.T).float())
            self.fc.layers[0].bias = torch.nn.Parameter(torch.from_numpy(np.zeros((1, 1))).float())
            out = self.fc.forward({'x': batch['input'].float()}).detach()
        return out

# class ARModel(nn.Module):
#     """
#     TODO
#     """
# 
#     def __init__(self, dims, order, increment, base_start=0):
#         """
#         Args:
#             dims (List[int]): input dim, hidden layer dims, and output dim
#                 TODO: right now can only handle input_dim = output_dim = 1
#             order (int): order of the AR model
#             increment (float): fixed interval between points in the domain
#         """
#         super().__init__()
#         self.fc = FCModel(dims)
#         self.increment = increment
#         self.base_start = base_start
#         self.base_end = base_start + order*self.increment
#         self.base_case_y = torch.nn.Parameter(torch.randn(order))
# 
#     def forward(self, x):
#         x_np = x.detach().cpu().numpy()
#         if x_np >= self.base_start and x_np < self.base_end:
#             idx = (x_np - self.base_start) / self.increment
#             return self.base_case_y[idx]
#         num_steps = ((x_np - self.base_end) / self.increment) + 1
#         prev = self.base_case_y
#         for i in range(num_steps):
#             out = self.fc(prev)
#             prev = torch.cat([prev[1:], out], 0)
#         return out


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, lstm=False, use_x=False):
        super(RNNModel, self).__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # RNN
        self.lstm = lstm
        if lstm:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # use x values instead of constant increment
        self.use_x = use_x

    def forward(self, batch):
        if self.use_x:
            x = batch['x'].float()
        else:
            x = batch['input'].float()

        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim))

        #LSTM
        if self.lstm:
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)) 
            out, hn = self.rnn(x, (h0, c0))
        else:
            # One time step
            out, hn = self.rnn(x, h0)

        out = self.fc(out)
        return out
