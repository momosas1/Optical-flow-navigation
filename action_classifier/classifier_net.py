import torch
import torch.nn as nn
from util.utils import Flatten


class Classifier(nn.Module):
    '''
    An action classifier network
    input: two stacked flow pictures (6*256*256)
    output: action probability
    '''

    def __init__(self):
        super().__init__()

        self.cnn_flow = self._init_flow_model()
        self.layer_init()

    def _init_flow_model(self):
        '''
        change the n_input_flow to control the input size
        use two flow picture: 2 * 3 channels
        '''
        self._n_input_flow = 6

        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        # FIX

        return nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_flow,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            Flatten(),
            nn.Linear(32 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512,4),
        )

    def layer_init(self):
        for layer in self.cnn_flow:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                nn.init.constant_(layer.bias, val=0)

    def forward(self, observations):
        #return propability

        action = self.cnn_flow(observations)
        return action
