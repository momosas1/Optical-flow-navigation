import torch
import torch.nn as nn

from util.utils import Flatten


class Net_with_flow(nn.Module):
    """
    Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size):
        super().__init__()

        self._n_input_goal = 2
        self._hidden_size = hidden_size

        # two same cnn
        self.cnn = self._init_perception_model(observation_space)
        self.cnn_flow = self._init_flow_model(observation_space)
        self.action_embedding = nn.Embedding(4, 32)
        self.rnn = nn.GRU(
            self._hidden_size + self._n_input_goal + 32, self._hidden_size
        )

        #get values of action
        self.critic_linear = nn.Linear(self._hidden_size, 1)
        self.layer_init()

    def _init_flow_model(self, observation_space):
        '''
        flow encoder
        use the net to predict action probability and then turn it to label and embedding
        '''
        self._n_input_flow = 6

        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        net = nn.Sequential(
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
            nn.Linear(32 * 28 * 28, self._hidden_size),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

        return net

    def _init_perception_model(self, observation_space):
        '''
        rgbd encoder
        '''
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0
        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        net = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_rgb + self._n_input_depth,
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
            nn.Linear(32 * 28 * 28, self._hidden_size),
            nn.ReLU(),
        )
        return net


    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                nn.init.constant_(layer.bias, val=0)

        for layer in self.cnn_flow:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                nn.init.constant_(layer.bias, val=0)

        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        nn.init.orthogonal_(self.critic_linear.weight, gain=1)
        nn.init.constant_(self.critic_linear.bias, val=0)

    def forward_rnn(self, x, hidden_states, masks):
        if x.size(0) == hidden_states.size(0):
            x, hidden_states = self.rnn(
                x.unsqueeze(0), (hidden_states * masks).unsqueeze(0)
            )
            x = x.squeeze(0)
            hidden_states = hidden_states.squeeze(0)
        else:
            n = hidden_states.size(0)
            t = int(x.size(0) / n)
            x = x.view(t, n, x.size(1))
            masks = masks.view(t, n)
            has_zeros = (
                (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
            )
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            has_zeros = [0] + has_zeros + [t]
            hidden_states = hidden_states.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                rnn_scores, hidden_states = self.rnn(
                    x[start_idx:end_idx],
                    hidden_states * masks[start_idx].view(1, -1, 1),
                )
                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0)
            x = x.view(t * n, -1)
            hidden_states = hidden_states.squeeze(0)

        return x, hidden_states

    def forward_perception_model(self, observations):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            cnn_input.append(rgb_observations)
        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        cnn_input = torch.cat(cnn_input, dim=1)
        # get rgbd feature
        rgbd = self.cnn(cnn_input)

        return rgbd

    def forward_flow_model(self, observations):
        flow_input = []

        flow_observations = observations["flow"]
        flow_observations = flow_observations.permute(0, 3, 1, 2)
        flow_input.append(flow_observations)

        pre_flow_observations = observations["pre_flow"]
        pre_flow_observations = pre_flow_observations.permute(0, 3, 1, 2)
        flow_input.append(pre_flow_observations)

        flow_input = torch.cat(flow_input, dim=1)
        flow = self.cnn_flow(flow_input)
        return flow

    def forward(self, observations, rnn_hidden_states, masks):
        x = observations["sensor"]
        rgbd = self.forward_perception_model(observations)
        pre_action = self.forward_flow_model(observations)

        #turn action probabilty to label
        _, predicted = torch.max(pre_action, 1)
        #turn label to embedding
        action = self.action_embedding((predicted.float() * masks).long().squeeze(dim=-1))
        if masks.shape[0] == 500:
            action = action[0]

        x = torch.cat([rgbd, x, action], dim=1)
        x, rnn_hidden_states = self.forward_rnn(x, rnn_hidden_states, masks)
        return self.critic_linear(x), x, rnn_hidden_states, predicted