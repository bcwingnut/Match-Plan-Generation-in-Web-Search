import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn
import numpy.ma as ma
import numpy as np
import copy
import datetime
import csv
import codecs
import pickle
import os

from Experiments_RL.experiment_base import HybridBase
from Utils_RL.models import *
from Utils_RL.utils import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PASAC_Agent_ERO_Continuous(HybridBase):
    def __init__(self, env, debug, weights, gamma, replay_buffer_size, max_steps,
                 hidden_size, value_lr, policy_lr, batch_size, state_dim,
                 action_discrete_dim, action_continuous_dim, rw_scale, soft_tau=1e-2, 
                 use_exp=True, discrete_log_prob_scale=10):
        super(PASAC_Agent_ERO_Continuous, self).__init__(debug, weights, gamma, replay_buffer_size, max_steps,
                                        hidden_size, value_lr, policy_lr, batch_size, state_dim,
                                             action_discrete_dim, action_continuous_dim)

        self.env = env
        
        self.replay_buffer = ReplayBuffer_ERO(replay_buffer_size, device, rw_scale, replay_updating_step=self.debug['replay_updating_step'])
        
        # [constants] copy

        self.soft_q_net1 = PASAC_QNetwork_MLP(max_steps, state_dim, action_discrete_dim, action_continuous_dim,
                                             hidden_size=hidden_size, batch_size=batch_size).to(self.device)
        self.soft_q_net2 = PASAC_QNetwork_MLP(max_steps, state_dim, action_discrete_dim, action_continuous_dim,
                                             hidden_size=hidden_size, batch_size=batch_size).to(self.device)
        self.target_soft_q_net1 = PASAC_QNetwork_MLP(max_steps, state_dim, action_discrete_dim, action_continuous_dim,
                                                    hidden_size=hidden_size, batch_size=batch_size).to(self.device)
        self.target_soft_q_net2 = PASAC_QNetwork_MLP(max_steps, state_dim, action_discrete_dim, action_continuous_dim,
                                                    hidden_size=hidden_size, batch_size=batch_size).to(self.device)
        self.policy_net = PASAC_PolicyNetwork_Continuous_MLP(state_dim,
                                                 max_steps,
                                                 action_continuous_dim,
                                                 hidden_size=hidden_size,
                                                 batch_size=batch_size).to(self.device)
        self.log_alpha_c = torch.zeros(
            1, dtype=torch.float32, requires_grad=True, device=self.device)

        # print('Soft Q Network (1,2): ', self.soft_q_net1)
        # print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.models = {'policy': self.policy_net,  'score': self.replay_buffer.score_net,
                       'value1': self.soft_q_net1, 'target_value1': self.target_soft_q_net1,
                       'value2': self.soft_q_net2, 'target_value2': self.target_soft_q_net2,}

        alpha_lr = 3e-4

        self.soft_q_optimizer1 = torch.optim.Adam(
            self.soft_q_net1.parameters(), lr=value_lr, weight_decay=self.debug['L2_norm'])
        self.soft_q_optimizer2 = torch.optim.Adam(
            self.soft_q_net2.parameters(), lr=value_lr, weight_decay=self.debug['L2_norm'])
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=policy_lr, weight_decay=self.debug['L2_norm'])
        self.alpha_optimizer_c = torch.optim.Adam(
            [self.log_alpha_c], lr=alpha_lr, weight_decay=self.debug['L2_norm'])

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.policy_param_noise = 0.3
        self.noise_clip = 0.5
        self.max_steps = max_steps
        self.DETERMINISTIC = False
        self.soft_tau = soft_tau
        self.use_exp = use_exp
        self.discrete_log_prob_scale = discrete_log_prob_scale
        self.alpha_c = 1.

        if self.debug['env'] == 'Reacher-v2':
            self.w_policy_loss = 0.1
        if self.debug['env'] == 'HalfCheetah-v2':
            self.w_policy_loss = 1

        # load models if needed
        if self.debug['load_model'] and self.debug['load_filename'] is not None:
            self.load(None)

        if self.debug['load_model'] and self.debug['replay_buffer_save_path'] is not None:
            df = open(self.debug['replay_buffer_save_path'], 'rb')
            self.replay_buffer=pickle.load(df)
            df.close()
            self.replay_buffer=self.replay_buffer.uncompress()

    def act(self, state: np.ndarray, sampling: bool = False, need_print=False):
        """
        explore with original action and return one-hot action encoding
        Note - `sampling` = sample an action tuple with the probability = <normalized action embedding>
        """

        action = self.policy_net.get_action(state, deterministic=self.DETERMINISTIC)
        return action

    def update(self, batch_size, episode=0, auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2, need_print=False):
        episodes = None
        indices = np.random.choice(len(self.subset_indices), batch_size)
        state, _, param, reward, next_state, done, episodes = self.subset_buffer[0][indices],self.subset_buffer[1][indices],self.subset_buffer[2][indices],self.subset_buffer[3][indices],self.subset_buffer[4][indices],self.subset_buffer[5][indices],self.subset_buffer[6][indices]
        episodes = torch.FloatTensor(np.array(episodes)).unsqueeze(-1).unsqueeze(-1).to(self.device)
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        param      = torch.FloatTensor(param).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        # [predict]
        predicted_q_value1 = self.soft_q_net1(state, None, param)
        predicted_q_value2 = self.soft_q_net2(state, None, param)
        new_action, log_prob, _, _, _ = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)

        # TODO: debug action_mean_log_prob
        

        if auto_entropy:

            alpha_loss_c = -(self.log_alpha_c * (log_prob +
                                             target_entropy).detach())
            alpha_loss_c = alpha_loss_c.mean()
            alpha_loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)
            self.alpha_optimizer_c.step()
            self.alpha_c = self.log_alpha_c.exp()
        else:
            self.alpha_c = 1.
            alpha_loss_c = 0
        if need_print:
            print('[debug: alpha_c]', self.alpha_c.data[0].item())

        # [compute value loss]
        predict_target_q1 = self.target_soft_q_net1(
            next_state, [], new_next_action)
        predict_target_q2 = self.target_soft_q_net2(
            next_state, [], new_next_action)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - \
            self.alpha_c * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min

        q_value_loss1_elementwise = predicted_q_value1 - target_q_value.detach()
        q_value_loss2_elementwise = predicted_q_value2 - target_q_value.detach()

        # [compute policy loss]
        predict_q1 = self.soft_q_net1(state, [], new_action)
        predict_q2 = self.soft_q_net2(state, [], new_action)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)

        policy_loss_elementwise = self.alpha_c * log_prob - predicted_new_q_value

        total_errors = (q_value_loss1_elementwise**2+q_value_loss2_elementwise**2+policy_loss_elementwise**2*self.w_policy_loss).detach().cpu().numpy().squeeze(1)
        q_value_loss1 = (q_value_loss1_elementwise**2).mean()
        q_value_loss2 = (q_value_loss2_elementwise**2).mean()
        policy_loss = (policy_loss_elementwise).mean()
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), 5)
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), 5)
        self.soft_q_optimizer2.step()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)
        self.policy_optimizer.step()

        #print('q loss: ', q_value_loss1, q_value_loss2)
        #print('policy loss: ', policy_loss )

        self.replay_buffer.score_update(self.subset_indices[indices], total_errors, episode)

        # -----Soft update the target value net-----
        soft_update(self.target_soft_q_net1, self.soft_q_net1, soft_tau)
        soft_update(self.target_soft_q_net2, self.soft_q_net2, soft_tau)

        return predicted_new_q_value.mean(), total_errors.mean()

    def collect_demonstration(self, dem_number, need_print: bool=False):
        if self.debug['load_demonstration']:
            df = open(self.debug['demonstration_path'], 'rb')
            demonstrations = pickle.load(df)
            df.close()
        else:
            if self.debug['env'] == 'Platform-v0':
                base_model_path='base_model/0212-benchmark-sac-mlp-Platform-v0-nni_param-1_490000_Feb_15_2020_21_42_35_policy.pt'
                base_max_steps=10
                base_hidden_size=1024
            elif self.debug['env'] == 'Goal-v0':
                base_model_path='base_model/0212-benchmark-sac-mlp-Goal-v0-nni_param_80000_Feb_13_2020_06_13_50_policy.pt'
                base_max_steps=10
                base_hidden_size=2048
            base_actor = SAC_PolicyNetwork_MLP(self.env.observation_space.shape[0],
                                                 base_max_steps,
                                                 self.env.action_space.spaces[0].n,
                                                 self.env.action_space.spaces[1].shape[0],
                                                 base_hidden_size
                                                 ).to(self.device)
            base_actor.load_state_dict(torch.load(base_model_path))
            demonstrations = []
            state = self.env.reset()
            for idx in range(dem_number):
                # -----init-----
                param, action_v = base_actor.get_action(state, deterministic=True)
                action_enc = (action_v, param)
                action = v2id((action_v, param), from_tensor=False)
                next_state, reward, done, info = self.env.step(action)
                demonstrations.append((state, action_v, param, reward, next_state, done, 0))
                state = next_state
                if done:
                    state = self.env.reset()
                if idx%100==0:
                    print('Collected {} demonstrations'.format(idx))

        if self.debug['save_demonstration']:
            print('saving demonstration to', self.debug['demonstration_path'])
            df = open(self.debug['demonstration_path'], 'wb')
            pickle.dump(demonstrations, df)
            df.close()

    def update_replay_policy(self, current_cumulative_reward, previous_cumulative_reward, episode):
        self.subset_buffer, self.subset_indices, avg_replay_loss = self.replay_buffer.policy_update(self.batch_size, current_cumulative_reward, previous_cumulative_reward, episode)
        return avg_replay_loss

def is_prioritized(buffer):
    return isinstance(buffer, PrioritizedReplayBuffer_SAC_MLP) or \
        isinstance(buffer, PrioritizedReplayBuffer_SAC3_MLP) or \
        isinstance(buffer, PrioritizedReplayBuffer_Original) or \
        isinstance(buffer, PrioritizedReplayBuffer_SAC2_MLP)

def has_multiple_bins(buffer):
    return isinstance(buffer, PrioritizedReplayBuffer_SAC2_MLP) or \
        isinstance(buffer, PrioritizedReplayBuffer_SAC3_MLP)