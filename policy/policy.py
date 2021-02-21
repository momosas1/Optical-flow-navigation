import torch
import torch.nn as nn
from policy.network import Net_with_flow
from util.utils import  CategoricalNet,DiagGaussian


class Policy_flow_fusion(nn.Module):
    '''
    the policy network
    use current rgbd and last action flow to predict the new action
    '''
    def __init__(self, observation_space, action_space, hidden_size=512):
        super().__init__()
        #self.dim_actions = 2
        self.dim_actions = 4

        self.net = Net_with_flow(
            observation_space=observation_space, hidden_size=hidden_size
        )
        self.action_distribution = CategoricalNet(self.net.output_size, self.dim_actions)


        #self.action_distribution = DiagGaussian(self.net.output_size, self.dim_actions)


    def forward(self, *x):
        raise NotImplementedError

    def act(self, observations, rnn_hidden_states, masks, deterministic=False):
        #sample action
        value, actor_features, rnn_hidden_states, pre_action = self.net(
            observations, rnn_hidden_states, masks
        )
        distribution = self.action_distribution(actor_features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, pre_action

    def get_value(self, observations, rnn_hidden_states, masks):
        #get the value of the action

        value, _, _, _ = self.net(observations, rnn_hidden_states, masks)
        return value

    def evaluate_actions(self, observations, rnn_hidden_states, masks, action):
        #calculate the param of ppo

        value, actor_features, rnn_hidden_states, pre_action = self.net(
            observations, rnn_hidden_states, masks
        )
        distribution = self.action_distribution(actor_features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states