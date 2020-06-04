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


class PASAC_Agent_DEM_MLP(HybridBase):
    def __init__(self, env, debug, weights, gamma, replay_buffer_size, max_steps,
                 hidden_size, value_lr, policy_lr, batch_size, state_dim,
                 action_discrete_dim, action_continuous_dim, soft_tau=1e-2, 
                 use_exp=True, discrete_log_prob_scale=10, reward_l=0, reward_h=1, data_bin=8):
        super(PASAC_Agent_DEM_MLP, self).__init__(debug, weights, gamma, replay_buffer_size, max_steps,
                                        hidden_size, value_lr, policy_lr, batch_size, state_dim,
                                             action_discrete_dim, action_continuous_dim)

        self.env = env
        
        assert debug['replay_buffer'] in ['n', 'p', 's', 'l', 'lp', 'ld', 'lp2', 'lp3', 'op']
        if debug['replay_buffer'] == 'n':
            self.replay_buffer = ReplayBuffer(replay_buffer_size)
        elif debug['replay_buffer'] == 'p':
            self.replay_buffer = PrioritizedReplayBuffer(replay_buffer_size)
        elif debug['replay_buffer'] == 's':
            self.replay_buffer = RewardStratifiedReplayBuffer(replay_buffer_size)
        elif debug['replay_buffer'] == 'l':
            self.replay_buffer = ReplayBufferLSTM2(replay_buffer_size)
        elif debug['replay_buffer'] == 'lp':
            self.replay_buffer = PrioritizedReplayBuffer_SAC_MLP(replay_buffer_size)
        elif debug['replay_buffer'] == 'lp2':
            self.replay_buffer = PrioritizedReplayBuffer_SAC2_MLP(replay_buffer_size, reward_l, reward_h, data_bin)
        elif debug['replay_buffer'] == 'lp3':
            self.replay_buffer = PrioritizedReplayBuffer_SAC3_MLP(replay_buffer_size, debug['capacity_distribution'], reward_l, reward_h, data_bin)
        elif debug['replay_buffer'] == 'op':
            self.replay_buffer = PrioritizedReplayBuffer_Original(replay_buffer_size)
        
        # [constants] copy

        self.soft_q_net1 = PASAC_QNetwork_MLP(max_steps, state_dim, action_discrete_dim, action_continuous_dim,
                                             hidden_size=hidden_size, batch_size=batch_size).to(self.device)
        self.soft_q_net2 = PASAC_QNetwork_MLP(max_steps, state_dim, action_discrete_dim, action_continuous_dim,
                                             hidden_size=hidden_size, batch_size=batch_size).to(self.device)
        self.target_soft_q_net1 = PASAC_QNetwork_MLP(max_steps, state_dim, action_discrete_dim, action_continuous_dim,
                                                    hidden_size=hidden_size, batch_size=batch_size).to(self.device)
        self.target_soft_q_net2 = PASAC_QNetwork_MLP(max_steps, state_dim, action_discrete_dim, action_continuous_dim,
                                                    hidden_size=hidden_size, batch_size=batch_size).to(self.device)
        self.policy_net = PASAC_PolicyNetwork_MLP(state_dim,
                                                 max_steps,
                                                 action_discrete_dim,
                                                 action_continuous_dim,
                                                 hidden_size=hidden_size,
                                                 batch_size=batch_size).to(self.device)
        self.log_alpha_c = torch.zeros(
            1, dtype=torch.float32, requires_grad=True, device=self.device)
        self.log_alpha_d = torch.tensor(
            [-1.6094], dtype=torch.float32, requires_grad=True, device=self.device)

        # print('Soft Q Network (1,2): ', self.soft_q_net1)
        # print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.models = {'policy': self.policy_net,
                       'value1': self.soft_q_net1, 'target_value1': self.target_soft_q_net1,
                       'value2': self.soft_q_net2, 'target_value2': self.target_soft_q_net2}

        alpha_lr = 3e-4

        self.soft_q_optimizer1 = torch.optim.Adam(
            self.soft_q_net1.parameters(), lr=value_lr, weight_decay=self.debug['L2_norm'])
        self.soft_q_optimizer2 = torch.optim.Adam(
            self.soft_q_net2.parameters(), lr=value_lr, weight_decay=self.debug['L2_norm'])
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=policy_lr, weight_decay=self.debug['L2_norm'])
        self.alpha_optimizer_c = torch.optim.Adam(
            [self.log_alpha_c], lr=alpha_lr, weight_decay=self.debug['L2_norm'])
        self.alpha_optimizer_d = torch.optim.Adam(
            [self.log_alpha_d], lr=alpha_lr, weight_decay=self.debug['L2_norm'])

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.policy_param_noise = 0.3
        self.noise_clip = 0.5
        self.max_steps = max_steps
        self.DETERMINISTIC = False
        self.soft_tau = soft_tau
        self.use_exp = use_exp
        self.discrete_log_prob_scale = discrete_log_prob_scale

        # load models if needed
        if self.debug['load_model'] and self.debug['load_filename'] is not None:
            self.load(None)

        if self.debug['load_model'] and self.debug['replay_buffer_save_path'] is not None:
            df = open(self.debug['replay_buffer_save_path'], 'rb')
            self.replay_buffer=pickle.load(df)
            df.close()
            self.replay_buffer=self.replay_buffer.uncompress()
        
        if self.debug['use_demonstration']:
            if self.debug['demonstration_buffer']=='n':
                self.demonstration_buffer = ReplayBuffer_MLP(self.debug['demonstration_buffer_size'])
            elif self.debug['demonstration_buffer']=='lp2':
                self.demonstration_buffer = PrioritizedReplayBuffer_SAC2_MLP(self.debug['demonstration_buffer_size'], reward_l, reward_h, data_bin)
            elif self.debug['demonstration_buffer']in['lp3','ld']:
                self.demonstration_buffer = PrioritizedReplayBuffer_SAC3_MLP(self.debug['demonstration_buffer_size'], 'uniform', reward_l, reward_h, data_bin)
            elif self.debug['demonstration_buffer']=='op':
                self.demonstration_buffer =  PrioritizedReplayBuffer_Original(self.debug['demonstration_buffer_size'])

            self.collect_demonstration(self.debug['demonstration_number'])
        else:
            self.demonstration_buffer = None
        self.demonstration_ratio = lambda x: max(0, 1-x*self.debug['demonstration_ratio_step']) if self.debug['use_demonstration'] else 0

    def act(self, state: np.ndarray, sampling: bool = False, need_print=False):
        """
        explore with original action and return one-hot action encoding
        Note - `sampling` = sample an action tuple with the probability = <normalized action embedding>
        """

        param, action_v = self.policy_net.get_action(
            state, deterministic=self.DETERMINISTIC)
        action_enc = (action_v, param)
        if sampling:  # TODO sampling as a categorical distribution
            a = np.log(action_v) / 1.0
            dist = np.exp(a) / np.sum(np.exp(a))
            choices = range(len(a))
            action = np.random.choice(choices, p=dist), param
        else:
            action = v2id((action_v, param), from_tensor=False)
        return action, action_enc, action_v, param

    def update(self, batch_size, episode=0, auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2, need_print=False):
        episodes = None
        indices, dem_indices = [], []
        if self.debug['use_demonstration'] and self.debug['demonstration_buffer']=="ld":
            dynamic_demonstration_num=(batch_size/self.demonstration_buffer.data_bin*np.maximum(1-2*(np.array(self.replay_buffer.bin_size)/np.array(self.demonstration_buffer.bin_size)),0)).astype(np.int)
            demonstration_num = sum(dynamic_demonstration_num)
            if need_print: print('Demonstrations sampled from each bin:'+str(list(dynamic_demonstration_num)))
        elif self.debug['pretrain']:
            demonstration_num = batch_size if episode<self.debug['pretrain_episodes'] else 0
        else:
            demonstration_num = int(batch_size*self.demonstration_ratio(episode))
        weights = None
        if self.debug['use_demonstration'] and self.debug['demonstration_buffer']=="ld":
            batch, indices, weights = self.replay_buffer.sample(
                batch_size//self.replay_buffer.data_bin-dynamic_demonstration_num)
            state, action_v, param, reward, next_state, done, episodes = batch
        elif isinstance(self.replay_buffer, ReplayBufferLSTM2):
            state, action_v, param, reward, next_state, done = self.replay_buffer.sample(
                batch_size-demonstration_num)
        elif is_prioritized(self.replay_buffer):
            batch, indices, weights = self.replay_buffer.sample(batch_size-demonstration_num)
            state, action_v, param, reward, next_state, done, episodes = batch
        else:
            state, action_v, param, reward, next_state, done, episodes = self.replay_buffer.sample(batch_size-demonstration_num)
        if self.debug['use_demonstration']:
            if self.debug['demonstration_buffer']=="ld":
                dem_batch, dem_indices, dem_weights = self.demonstration_buffer.sample(dynamic_demonstration_num)
                dem_state, dem_action_v, dem_param, dem_reward, dem_next_state, dem_done, dem_episodes = dem_batch
            elif self.debug['demonstration_buffer'] in ['n'] and demonstration_num>0:
                dem_state, dem_action_v, dem_param, dem_reward, dem_next_state, dem_done, dem_episodes = self.demonstration_buffer.sample(demonstration_num)
                dem_indices = [-1] * demonstration_num
                dem_weights = [0] * demonstration_num
            else:
                dem_batch, dem_indices, dem_weights = self.demonstration_buffer.sample(demonstration_num)
                dem_state, dem_action_v, dem_param, dem_reward, dem_next_state, dem_done, dem_episodes = dem_batch
            state = state + dem_state
            action_v = action_v + dem_action_v
            param = param + dem_param
            reward = reward + dem_reward
            next_state = next_state + dem_next_state
            done = done + dem_done
            indices = indices + dem_indices
            weights = np.concatenate((weights, dem_weights))
            dem_mask = torch.FloatTensor(np.concatenate((np.zeros(len(indices)-len(dem_indices)),np.ones(len(dem_indices))))).to(self.device)
            episodes = episodes+dem_episodes
        if episodes is not None:
            episodes = torch.FloatTensor(np.array(episodes)).to(self.device)
        if weights is not None:
            weights /= max(weights)
            if self.debug['demonstration_buffer'] in ['n']:
                weights[batch_size-demonstration_num:]=0.5
            weights = torch.FloatTensor(weights).unsqueeze(-1).to(self.device)


        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action_v   = torch.FloatTensor(action_v).to(self.device)
        param      = torch.FloatTensor(param).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        # [predict]
        predicted_q_value1 = self.soft_q_net1(state, action_v, param)
        predicted_q_value2 = self.soft_q_net2(state, action_v, param)
        new_action_v, new_param, log_prob, _, _, _ = self.policy_net.evaluate(state)
        new_next_action_v, new_next_param, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)

        # TODO: debug action_mean_log_prob
        def get_log_action_prob(action_prob, use_exp):
            if self.use_exp:  # expectation
                action_log_prob = torch.log(action_prob)
                action_log_prob = action_log_prob.mul(action_prob)  # calculate expectation
                action_log_prob[action_log_prob!=action_log_prob] = 0  # set NaN to zero
                action_log_prob.clamp_(-10, 0)
                action_sum_log_prob = torch.sum(action_log_prob, dim=-1)
                action_sum_log_prob = action_sum_log_prob.view(action_sum_log_prob.size(0), 1)
            else:  # sampling
                action_sample_all = []
                for action in action_prob:
                    action_sample = []
                    for a in action:
                        a = a.detach().cpu().numpy()
                        choices = range(len(a))
                        if a[0] == 0:
                            action_sample.append(0)
                            continue
                        idx = np.random.choice(choices, p=a)
                        action_sample.append(a[idx].item())
                    action_sample_all.append(action_sample)
                action_sample_all = torch.FloatTensor(action_sample_all)
                action_log_prob = torch.log(action_sample_all)
                action_sum_log_prob = action_log_prob.view(action_log_prob.size(0),
                                                           self.max_steps, 1)
                action_sum_log_prob = action_sum_log_prob.to(self.device)

            return action_sum_log_prob

        action_sum_log_prob = get_log_action_prob(new_action_v, self.use_exp)
        action_sum_log_prob_next = get_log_action_prob(new_next_action_v, self.use_exp)

        if auto_entropy:
            alpha_loss_d = -(self.log_alpha_d * (action_sum_log_prob +
                                             target_entropy).detach())
            alpha_loss_d = alpha_loss_d.mean()
            self.alpha_optimizer_d.zero_grad()
            alpha_loss_d.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)
            self.alpha_optimizer_d.step()
            self.alpha_d = self.log_alpha_d.exp()

            alpha_loss_c = -(self.log_alpha_c * (log_prob +
                                             target_entropy).detach())
            alpha_loss_c = alpha_loss_c.mean()
            self.alpha_optimizer_d.zero_grad()
            alpha_loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)
            self.alpha_optimizer_c.step()
            self.alpha_c = self.log_alpha_c.exp()
        else:
            self.alpha_c = 1.
            self.alpha_d = 1.
            alpha_loss_d = 0
            alpha_loss_c = 0
        if need_print:
            print('[debug: alpha_c]', self.alpha_c.data[0].item())
            print('[debug: alpha_d]', self.alpha_d.data[0].item())

        # [compute value loss]
        predict_target_q1 = self.target_soft_q_net1(
            next_state, new_next_action_v, new_next_param)
        predict_target_q2 = self.target_soft_q_net2(
            next_state, new_next_action_v, new_next_param)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - \
            self.alpha_c * next_log_prob - self.alpha_d * action_sum_log_prob_next
        target_q_value = reward + (1 - done) * gamma * target_q_min

        q_value_loss1_elementwise = predicted_q_value1 - target_q_value.detach()
        q_value_loss2_elementwise = predicted_q_value2 - target_q_value.detach()

        # [compute policy loss]
        predict_q1 = self.soft_q_net1(state, new_action_v, new_param)
        predict_q2 = self.soft_q_net2(state, new_action_v, new_param)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)

        policy_loss_elementwise = self.alpha_c * log_prob + \
            self.alpha_d * action_sum_log_prob - predicted_new_q_value

        if self.debug['use_demonstration'] and self.debug['behavior_cloning']:
            q_filter = ((predicted_q_value1 + predicted_q_value2)/2 > reward).squeeze(1)*dem_mask
            continuous_bcloss = ((new_param - param)**2).mean(dim=-1)
            discrete_bcloss = F.binary_cross_entropy(new_action_v, action_v, reduce=False).mean(dim=-1)
            behavior_cloning_loss_elementwise = q_filter * (continuous_bcloss + discrete_bcloss * self.debug["discrete_bcloss_weight"])
            policy_loss_elementwise += behavior_cloning_loss_elementwise.unsqueeze(-1) * self.debug['bcloss_weight']

        if is_prioritized(self.replay_buffer):
            # print((q_value_loss1_elementwise**2).mean().item(),(policy_loss_elementwise**2).mean().item(),(episode-episodes).mean().item())
            td = (q_value_loss1_elementwise.abs()+q_value_loss2_elementwise.abs()+policy_loss_elementwise*self.debug['policy_loss_w']).sum(dim=1)
            # print('qloss:',q_value_loss1_elementwise[0][0].item(), 'ploss',policy_loss_elementwise[0][0].item())
            priorities = td+(episode-episodes)*self.debug['ep_punishment']
            q_value_loss1 = (q_value_loss1_elementwise**2*weights).mean()
            q_value_loss2 = (q_value_loss2_elementwise**2*weights).mean()
            policy_loss = (policy_loss_elementwise*weights).mean()
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
        else:
            q_value_loss1 = self.soft_q_criterion1(q_value_loss1_elementwise, torch.zeros_like(
                q_value_loss1_elementwise))  # (q_value_loss1_elementwise**2).mean()
            q_value_loss2 = self.soft_q_criterion2(q_value_loss2_elementwise, torch.zeros_like(
                q_value_loss2_elementwise))  # (q_value_loss2_elementwise**2).mean()
            policy_loss = policy_loss_elementwise.mean()
            self.soft_q_optimizer1.zero_grad()
            q_value_loss1.backward()
            torch.nn.utils.clip_grad_norm_(
                self.soft_q_net1.parameters(), 5)
            self.soft_q_optimizer1.step()
            self.soft_q_optimizer2.zero_grad()
            q_value_loss2.backward()
            torch.nn.utils.clip_grad_norm_(
                self.soft_q_net2.parameters(), 5)
            self.soft_q_optimizer2.step()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), 5)
            self.policy_optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        if is_prioritized(self.replay_buffer):
            self.replay_buffer.priority_update(
                indices[:len(indices)-len(dem_indices)], priorities.tolist()[:len(indices)-len(dem_indices)], td.tolist()[:len(indices)-len(dem_indices)])
        if self.debug['use_demonstration'] and is_prioritized(self.demonstration_buffer):
            self.demonstration_buffer.priority_update(
                indices[len(indices)-len(dem_indices):], priorities.tolist()[len(indices)-len(dem_indices):], td.tolist()[len(indices)-len(dem_indices):])

        # -----Soft update the target value net-----
        soft_update(self.target_soft_q_net1, self.soft_q_net1, soft_tau)
        soft_update(self.target_soft_q_net2, self.soft_q_net2, soft_tau)

        return predicted_new_q_value.mean()

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
            base_actor = PASAC_PolicyNetwork_MLP(self.env.observation_space.shape[0],
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
        # -----add seq to replay buffer-----
        for experience in demonstrations:
            self.demonstration_buffer.push(*experience)
        # return self.reward_list
        if has_multiple_bins(self.demonstration_buffer):
                print('[Size of demonstration buffer]', str(self.demonstration_buffer.bin_size))

def is_prioritized(buffer):
    return isinstance(buffer, PrioritizedReplayBuffer_SAC_MLP) or \
        isinstance(buffer, PrioritizedReplayBuffer_SAC3_MLP) or \
        isinstance(buffer, PrioritizedReplayBuffer_Original) or \
        isinstance(buffer, PrioritizedReplayBuffer_SAC2_MLP)

def has_multiple_bins(buffer):
    return isinstance(buffer, PrioritizedReplayBuffer_SAC2_MLP) or \
        isinstance(buffer, PrioritizedReplayBuffer_SAC3_MLP)