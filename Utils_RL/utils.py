import copy
import random
from collections import namedtuple

import gym
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from Utils_RL.sum_tree import MinSegmentTree, SumSegmentTree, SumTree
import pdb

device = torch.device('cuda')
d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATransition = namedtuple('PATransition', ('state', 'action', 'param', 'reward', 'next_state', 'done'))  # ,


########################## (Parameterized) Action Utilities ##########################


def v2id(action_enc: tuple, from_tensor: bool = False):
    return action_enc[0].argmax().item(), action_enc[1] if not from_tensor else \
        action_enc[1].squeeze(dim=0).detach().cpu().numpy()


def id2v(action: tuple, dim_action: int, return_tensor: bool = False):
    """ convert one action tuple from discrete action id to one-hot encoding """
    one_hot = np.zeros(dim_action)  # number of actions including special actions
    one_hot[action[0]] = 1
    return (torch.Tensor(one_hot) if return_tensor else one_hot,
            torch.Tensor(action[1]) if return_tensor else action[1])


def soft_update(target_net, source_net, soft_tau):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau)


def copy_param(target_net, source_net):  # or copy.deepcopy(source_net)
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(param.data)


def padding_and_convert(seq, max_length, is_action=False):
    """padding a batch of sequence to max_length, then convert it into torch.FloatTensor"""
    seq_dim_len = torch.LongTensor([len(s) for s in seq])
    if torch.is_tensor(seq):
        for i, s in enumerate(seq):
            if s.shape[0] < max_length:
                zero_arrays = torch.zeros(max_length - s.shape[0], s.shape[0]).cuda()
                s = torch.cat((s, zero_arrays), 0)
            seq[i] = s
    else:
        for i, s in enumerate(seq):
            s = np.array(s)
            if len(s) < max_length:
                zero_arrays = np.zeros((max_length - len(s), len(s[0])))
                s = np.concatenate((s, zero_arrays), 0)
            seq[i] = s
        seq = torch.FloatTensor(seq).to(d)
    return seq, seq_dim_len


def gen_mask(seqs, seq_dim_len, max_len):
    m = []
    for i, seq in enumerate(seqs):
        m1 = []
        for j in range(max_len):
            if j < seq_dim_len[i]:
                m1.append([1])
            else:
                m1.append([0])
        m.append(m1)
    return m


########################## Experience Replay Buffer ##########################


class ReplayBuffer_LSTM:
    """ 
    Replay buffer for agent with LSTM network additionally using previous action, can be used 
    if the hidden states are not stored (arbitrary initialization of lstm for training).
    And each sample contains the whole episode instead of a single step.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst = [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            state, action, last_action, reward, next_state, done = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)

        return s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class PrioritizedReplayBuffer_LSTM:
    """ 
    Prioritized Replay Buffer see 
    https://arxiv.org/pdf/1511.05952.pdf and https://cardwing.github.io/files/RL_course_report.pdf for details
    """
    e = 0.01
    alpha = 2.0
    beta = 1.0
    beta_increment_per_sampling = 0.01
    data_bin = 8

    def __init__(self, capacity):
        """ Prioritized experience replay buffer initialization."""
        self.capacity = capacity
        self.buffer = SumTree(capacity)
        self._max_priority = 1.0

    def _get_priority(self, error):
        """ get priority for TD error"""
        return (np.abs(error) + self.e) ** self.alpha

    def push(self, state, action, last_action, reward, next_state, done):
        """  push a sample into prioritized replay buffer"""
        self.buffer.add(self._max_priority, (state, action, last_action, reward, next_state, done))

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.
        Parameters
        ----------
        indices :
            list of sample indices
        """
        for idx, error in zip(indices, priorities):
            p = self._get_priority(error)
            self._max_priority = max(self._max_priority, p)
            self.buffer.update(idx, p)

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """
        batch = []
        indices = []
        priorities = []
        weights = []
        segment = self.buffer.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # make Reward Bin
        data_len = len(self.buffer)
        reward_sum = [(i, np.sum(sequence[3])) for i, sequence in enumerate(self.buffer[:data_len])]
        reward_sum = sorted(reward_sum, key=lambda x: x[1], reverse=True)
        # sort first, then sequentially split the sorted dataset into bins
        # sample Top batch_size//2 from reward
        for j in range(self.data_bin):
            count = 0
            # for every bin
            for i, reward in reward_sum[data_len // self.data_bin * j:data_len // self.data_bin * (j + 1)]:
                # get index, priority, data
                index = i - 1 + self.buffer.capacity
                priority = self.buffer.tree[index]
                data = self.buffer.data[i]
                # append to list
                priorities.append(priority)
                weights.append((1. / self.capacity / priority) ** self.beta if priority > 1e-16 else 0)  # clip
                indices.append(index)
                batch.append(data)
                count += 1
                if count >= batch_size // (2 * self.data_bin):
                    break
                # for every bin, fetch batch_size//(2 * self.data_bin) data items

        # random sample
        len_random = batch_size - len(indices)
        for i in range(len_random):
            # do not get zero from SumTree
            while True:
                a = segment * i
                b = segment * (i + 1)
                r = random.uniform(a, b)
                index, priority, data = self.buffer.get(r)
                if data:
                    break
            priorities.append(priority)
            weights.append((1. / self.capacity / priority) ** self.beta if priority > 1e-16 else 0)
            indices.append(index)
            batch.append(data)  
            # [batch_size, max_step, 6(state, action_v, param, reward, next_state, done), *]

        # Normalize for stability
        weights = np.array(weights)
        weights /= max(weights)

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst = [], [], [], [], [], []
        for sample in batch:
            state, action, last_action, reward, next_state, done = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)

        return (s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst), indices, weights

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer.data[key]


class ReplayBuffer_MLP:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action_v, param, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action_v, param, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action_v, param, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action_v, param, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer_MLP:
    """ 
    Prioritized Replay Buffer see 
    https://arxiv.org/pdf/1511.05952.pdf and https://cardwing.github.io/files/RL_course_report.pdf for details
    """
    e = 0.01
    alpha = 2.0
    beta = 1.0
    beta_increment_per_sampling = 0.01
    data_bin = 8

    def __init__(self, capacity):
        """ Prioritized experience replay buffer initialization."""
        self.capacity = capacity
        self.buffer = SumTree(capacity)
        self._max_priority = 1.0

    def _get_priority(self, error):
        """ get priority for TD error"""
        return (np.abs(error) + self.e) ** self.alpha

    def push(self, state, action, last_action, reward, next_state, done):
        """  push a sample into prioritized replay buffer"""
        self.buffer.add(self._max_priority, (state, action, last_action, reward, next_state, done))

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.
        Parameters
        ----------
        indices :
            list of sample indices
        """
        for idx, error in zip(indices, priorities):
            p = self._get_priority(error)
            self._max_priority = max(self._max_priority, p)
            self.buffer.update(idx, p)

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """
        batch = []
        indices = []
        priorities = []
        weights = []
        segment = self.buffer.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # make Reward Bin
        data_len = len(self.buffer)
        reward_sum = [(i, sequence[3]) for i, sequence in enumerate(self.buffer[:data_len])]
        reward_sum = sorted(reward_sum, key=lambda x: x[1], reverse=True)
        # sort first, then sequentially split the sorted dataset into bins
        # sample Top batch_size//2 from reward
        for j in range(self.data_bin):
            count = 0
            for i, reward in reward_sum[data_len // self.data_bin * j:data_len // self.data_bin * (j + 1)]:
                # get index, priority, data
                index = i - 1 + self.buffer.capacity
                priority = self.buffer.tree[index]
                data = self.buffer.data[i]
                # append to list
                priorities.append(priority)
                weights.append((1. / self.capacity / priority) ** self.beta if priority > 1e-16 else 0)  # clip
                indices.append(index)
                batch.append(data)
                count += 1
                if count >= batch_size // (2 * self.data_bin):
                    break
                # for every bin, fetch batch_size//(2 * self.data_bin) data items

        # random sample
        len_random = batch_size - len(indices)
        for i in range(len_random):
            # do not get zero from SumTree
            while True:
                a = segment * i
                b = segment * (i + 1)
                r = random.uniform(a, b)
                index, priority, data = self.buffer.get(r)
                if data:
                    break
            priorities.append(priority)
            weights.append((1. / self.capacity / priority) ** self.beta if priority > 1e-16 else 0)
            indices.append(index)
            batch.append(data)  
            # [batch_size, max_step, 6(state, action_v, param, reward, next_state, done), *]

        # Normalize for stability
        weights = np.array(weights)
        weights /= max(weights)

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst = [], [], [], [], [], []
        for sample in batch:
            state, action, last_action, reward, next_state, done = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)

        return (s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst), indices, weights

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer.data[key]


########################## Action Wrapper ##########################


class NormalizedHybridActions(gym.ActionWrapper):
    """ Action Normalization: just normalize 2nd element in action tuple (id, (params)) """
    def action(self, action: tuple) -> tuple:  # old gym's Tuple needs extra `.spaces`
        low = self.action_space.spaces[1].low
        high = self.action_space.spaces[1].high

        param = low + (action[1] + 1.0) * 0.5 * (high - low)
        param = np.clip(param, low, high)

        return action[0], param

    def reverse_action(self, action: tuple) -> tuple:
        low = self.action_space.spaces[1].low
        high = self.action_space.spaces[1].high

        param = 2 * (action[1] - low) / (high - low) - 1
        param = np.clip(param, low, high)

        return action[0], param

class ReplayBuffer_MLP:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action_v, param, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action_v, param, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action_v, param, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action_v, param, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action_v, param, reward, next_state, done, episode):
        """ push a transition tuple with parameterized action (action encoding & parameters) """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            state, action_v, param, reward, next_state, done, episode)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """
        if batch_size == 0:
            return [],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        # action is 2-tuple, stack separately
        _s, _a_v, _p, _r, _n, _d, _episode = zip(*batch)
        state, action_v, param, reward, next_state, done, episode = map(
            list, (_s, _a_v, _p, _r, _n, _d, _episode))
        return state, action_v, param, reward, next_state, done, episode

    def compress(self, semantic_length=100):
        compressed_buffer = copy.deepcopy(self)
        for episode in compressed_buffer:
            if episode == 0:
                break
            for i in range(len(episode[0])):
                if i == 0:
                    continue
                episode[0][i] = episode[0][i][:-semantic_length]
        return compressed_buffer

    def uncompress(self, semantic_length=100):
        uncompressed_buffer = copy.deepcopy(self)
        for episode in uncompressed_buffer:
            if episode == 0:
                break
            for i in range(len(episode[0])):
                if i == 0:
                    continue
                episode[0][i] = np.concatenate((episode[0][i], episode[0][0][-semantic_length:]))
        return uncompressed_buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer[key]


class SequenceReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action_v, param, reward, next_state, done, episode):
        """ push a transition tuple with parameterized action (action encoding & parameters) """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action_v, param, reward, next_state, done, episode)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """
        batch = [random.choice(self.buffer) for _ in range(batch_size)]
        return batch

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer[key]


class PrioritizedReplayBuffer:
    """ 
    Prioritized Replay Buffer see 
    https://arxiv.org/pdf/1511.05952.pdf and https://cardwing.github.io/files/RL_course_report.pdf for details
    """
    e = 0.01
    alpha = 0.5
    beta = 1.0
    beta_increment_per_sampling = 0.01
    data_bin = 8

    def __init__(self, capacity):
        """ Prioritized experience replay buffer initialization."""
        self.capacity = capacity
        self.buffer = SumTree(capacity)

    def _get_priority(self, error):
        """ get priority for TD error"""
        return (np.abs(error) + self.e) ** self.alpha

    def push(self, error, sample):
        """  push a sample into prioritized replay buffer"""
        p = self._get_priority(error)
        self.buffer.add(p, sample)

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for idx, error in zip(indices, priorities):
            if idx == -1:
                continue
            p = self._get_priority(error)
            self.buffer.update(idx, p)

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """

        batch = []
        indices = []
        priorities = []
        weights = []
        segment = self.buffer.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        def clip_small(priority):
            if priority > 1e-16:
                return (1./self.capacity/priority)**self.beta
            else:
                return 0

        # sample with strategy
        ## make Reward Bin
        data_len = len(self.buffer)
        reward_sum = [(i, np.sum(x[3] for x in sequence))
                      for i, sequence in enumerate(self.buffer[:data_len])]
        reward_sum = sorted(reward_sum, key=lambda x: x[1], reverse=True)
        # sort first, then sequentially split the sorted dataset into 8 bins
        ## sample Top batch_size//2 from reward
        for j in range(self.data_bin):
            count = 0
            # random_list = random.sample(range(data_len//self.data_bin*j, data_len//self.data_bin*(j+1)), data_len//self.data_bin)
            # for every bin
            for i, reward in reward_sum[data_len//self.data_bin*j:data_len//self.data_bin*(j+1)]:
                # get index, priority, data
                index = i - 1 + self.buffer.capacity
                priority = self.buffer.tree[index]
                data = self.buffer.data[i]
                # append to list
                priorities.append(priority)
                weights.append((1./self.capacity/priority) **
                               self.beta if priority > 1e-16 else 0)
                indices.append(index)
                batch.append(data)
                count += 1
                if count >= batch_size//(2 * self.data_bin):
                    break
                # for every bin, fetch batch_size//(2 * self.data_bin) data items

        # # random sample
        len_random = batch_size - len(indices)
        for i in range(len_random):
            # do not get zero from SumTree
            while True:
                a = segment * i
                b = segment * (i + 1)
                r = random.uniform(a, b)
                index, priority, data = self.buffer.get(r)
                if data:
                    break
            priorities.append(priority)
            weights.append((1./self.capacity/priority) **
                           self.beta if priority > 1e-16 else 0)
            indices.append(index)
            batch.append(data)
            # # To avoid duplicating
            # self.buffer.update(index, 0)

        # sampling_priorities = priorities / self.buffer.total()
        # sampling_priorities = sorted(sampling_priorities, reverse=True)
        # print("[Debug] First 20 Sample Priorities:", sampling_priorities[:20])

        # Revert priorities
        # for idx, p in zip(indices, priorities):
        #     self.buffer.update(idx, p)

        # Normalize for stability
        weights = np.array(weights)
        weights /= max(weights)

        return batch, indices, weights

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer.data[key]


class RewardStratifiedReplayBuffer:
    data_bin = 8

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, sample):
        """ push a transition tuple with parameterized action (action encoding & parameters) """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """
        batch = []

        # sample with stratified reward strategy
        data_len = len(self.buffer)
        reward_sum = [(i, sum(x[3] for x in sequence))
                      for i, sequence in enumerate(self.buffer[:data_len])]
        reward_sum = sorted(reward_sum, key=lambda x: x[1], reverse=True)
        # sort first, then sequentially split the sorted dataset into some strata(bin)
        for j in range(self.data_bin):
            random_list = random.sample(range(
                data_len//self.data_bin*j, data_len//self.data_bin*(j+1)), batch_size//self.data_bin)
            # random sampling from every stratum
            for i in random_list:
                data = self.buffer[i]
                # append to list
                batch.append(data)
                # for every bin, fetch batch_size // self.data_bin data items

        return batch


class ReplayBufferLSTM2:
    """ 
    Replay buffer for agent with LSTM network additionally using previous action, can be used 
    if the hidden states are not stored (arbitrary initialization of lstm for training).
    And each sample contains the whole episode instead of a single step.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst = [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            state, action, last_action, reward, next_state, done = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)

        return s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class PrioritizedReplayBuffer_SAC_MLP:
    """ 
    Prioritized Replay Buffer see 
    https://arxiv.org/pdf/1511.05952.pdf and https://cardwing.github.io/files/RL_course_report.pdf for details
    """
    e = 0.01
    d = 100
    alpha = 2.0
    beta = 1.0
    beta_increment_per_sampling = 0.01
    data_bin = 8

    def __init__(self, capacity):
        """ Prioritized experience replay buffer initialization."""
        self.capacity = capacity
        self.buffer = SumTree(capacity)
        self._max_priority = 1.0

    def _get_priority(self, error):
        """ get priority for TD error"""
        return (np.abs(error) + self.e + self.d) ** self.alpha

    def push(self, state, action, last_action, reward, next_state, done, episode):
        """  push a sample into prioritized replay buffer"""
        self.buffer.add(self._max_priority, (state, action, last_action, reward, next_state, done, episode))

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for idx, error in zip(indices, priorities):
            p = self._get_priority(error)
            self._max_priority = max(self._max_priority, p)
            self.buffer.update(idx, p)

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """
        batch = []
        indices = []
        priorities = []
        weights = []
        segment = self.buffer.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        def clip_small(priority):
            if priority > 1e-16:
                return (1. / self.capacity / priority) ** self.beta
            else:
                return 0

        # sample with strategy
        # make Reward Bin
        data_len = len(self.buffer)
        reward_sum = [(i, sequence[3]) for i, sequence in enumerate(self.buffer[:data_len])]
        reward_sum = sorted(reward_sum, key=lambda x: x[1], reverse=True)
        # sort first, then sequentially split the sorted dataset into 8 bins
        # sample Top batch_size//2 from reward
        for j in range(self.data_bin):
            count = 0
            # random_list = random.sample(range(data_len//self.data_bin*j, data_len//self.data_bin*(j+1)), data_len//self.data_bin)
            # for every bin
            for i, reward in reward_sum[data_len//self.data_bin*j:data_len//self.data_bin*(j+1)]:
                if count >= batch_size//(2 * self.data_bin) or len(batch)==batch_size:
                    break
                # get index, priority, data
                index = i - 1 + self.buffer.capacity
                priority = self.buffer.tree[index]
                data = self.buffer.data[i]
                # append to list
                priorities.append(priority)
                weights.append((1./self.capacity/priority) **
                               self.beta if priority > 1e-16 else 0)
                indices.append(index)
                batch.append(data)
                count += 1
                # for every bin, fetch batch_size//(2 * self.data_bin) data items
            if len(batch)==batch_size:
                    break
        # random sample
        len_random = batch_size - len(indices)
        for i in range(len_random):
            # do not get zero from SumTree
            while True:
                a = segment * i
                b = segment * (i + 1)
                r = random.uniform(a, b)
                index, priority, data = self.buffer.get(r)
                if data:
                    break
            priorities.append(priority)
            weights.append((1./self.capacity/priority) **
                           self.beta if priority > 1e-16 else 0)
            indices.append(index)
            batch.append(data)  # [batch_size, max_step, 6(state, action_v, param, reward, next_state, done), *]
            # # To avoid duplicating
            # self.buffer.update(index, 0)

        # sampling_priorities = priorities / self.buffer.total()
        # sampling_priorities = sorted(sampling_priorities, reverse=True)
        # print("[Debug] First 20 Sample Priorities:", sampling_priorities[:20])

        # Revert priorities
        # for idx, p in zip(indices, priorities):
        #     self.buffer.update(idx, p)

        # Normalize for stability
        weights = np.array(weights)
        #weights /= max(weights)

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, e_lst = [], [], [], [], [], [], []
        for sample in batch:
            state, action, last_action, reward, next_state, done, episode = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            e_lst.append(episode)

        return (s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, e_lst), indices, weights

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer.data[key]

    def compress(self, semantic_length=100):
        compressed_buffer = copy.deepcopy(self)
        for episode in compressed_buffer:
            if episode == 0:
                break
            for i in range(len(episode[0])):
                if i == 0:
                    continue
                episode[0][i] = episode[0][i][:-semantic_length]
        return compressed_buffer

    def uncompress(self, semantic_length=100):
        uncompressed_buffer = copy.deepcopy(self)
        for episode in uncompressed_buffer:
            if episode == 0:
                break
            for i in range(len(episode[0])):
                if i == 0:
                    continue
                episode[0][i] = np.concatenate((episode[0][i], episode[0][0][-semantic_length:]))
        return uncompressed_buffer

class PrioritizedReplayBuffer_Original:
    """ 
    Prioritized Replay Buffer see 
    https://arxiv.org/pdf/1511.05952.pdf and https://cardwing.github.io/files/RL_course_report.pdf for details
    """
    e = 0.01
    d = 100
    alpha = 2.0
    beta = 1.0
    beta_increment_per_sampling = 0.01

    def __init__(self, capacity):
        """ Prioritized experience replay buffer initialization."""
        self.capacity = capacity
        self.buffer = SumTree(capacity)
        self._max_priority = 1.0

    def _get_priority(self, error):
        """ get priority for TD error"""
        return (np.abs(error) + self.e + self.d) ** self.alpha

    def push(self, state, action, last_action, reward, next_state, done, episode):
        """  push a sample into prioritized replay buffer"""
        self.buffer.add(self._max_priority, (state, action, last_action, reward, next_state, done, episode))

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for idx, error in zip(indices, priorities):
            p = self._get_priority(error)
            self._max_priority = max(self._max_priority, p)
            self.buffer.update(idx, p)

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """
        batch = []
        indices = []
        priorities = []
        weights = []
        segment = self.buffer.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(batch_size):
            # do not get zero from SumTree
            while True:
                a = segment * i
                b = segment * (i + 1)
                r = random.uniform(a, b)
                index, priority, data = self.buffer.get(r)
                if data:
                    break
            priorities.append(priority)
            weights.append((1./self.capacity/priority) **
                           self.beta if priority > 1e-16 else 0)
            indices.append(index)
            batch.append(data)  # [batch_size, max_step, 6(state, action_v, param, reward, next_state, done), *]
            # # To avoid duplicating
            # self.buffer.update(index, 0)

        # sampling_priorities = priorities / self.buffer.total()
        # sampling_priorities = sorted(sampling_priorities, reverse=True)
        # print("[Debug] First 20 Sample Priorities:", sampling_priorities[:20])

        # Revert priorities
        # for idx, p in zip(indices, priorities):
        #     self.buffer.update(idx, p)

        # Normalize for stability
        weights = np.array(weights)
        #weights /= max(weights)

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst = [], [], [], [], [], [], []
        for sample in batch:
            state, action, last_action, reward, next_state, done, episode = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            episode_lst.append(episode)

        return (s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst), indices, weights

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer.data[key]

    def compress(self, semantic_length=100):
        compressed_buffer = copy.deepcopy(self)
        for episode in compressed_buffer:
            if episode == 0:
                break
            for i in range(len(episode[0])):
                if i == 0:
                    continue
                episode[0][i] = episode[0][i][:-semantic_length]
        return compressed_buffer

    def uncompress(self, semantic_length=100):
        uncompressed_buffer = copy.deepcopy(self)
        for episode in uncompressed_buffer:
            if episode == 0:
                break
            for i in range(len(episode[0])):
                if i == 0:
                    continue
                episode[0][i] = np.concatenate((episode[0][i], episode[0][0][-semantic_length:]))
        return uncompressed_buffer

class ReplayBuffer_ERO:
    
    """ 
    Experience Replay Optimization see 
    https://www.ijcai.org/proceedings/2019/589 for details
    """

    def __init__(self, capacity, device):
        """ Prioritized experience replay buffer initialization."""
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=object)
        self.scores = np.zeros(capacity)
        self.td = np.zeros(capacity)
        self.replay_updating_step = 1
        self.device = device
        self.score_net = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        ).to(device)
        self.position = 0
        self.size = 0
        self.score_net_optimizer = torch.optim.Adam(self.score_net.parameters(), lr=1e-4, weight_decay=0.01)
        self.replay_updating_batch_size = 64


    def push(self, state, action_v, param, reward, next_state, done, episode):
        """  push a sample into prioritized replay buffer"""
        self.size = min(self.size+1, self.capacity)
        self.buffer[self.position] = (state, action_v, param, reward, next_state, done, episode)
        self.scores[self.position] = 1
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def policy_update(self, batch_size, current_cummulative_reward, previous_cummulative_reward, current_episode):
        replay_reward = current_cummulative_reward - previous_cummulative_reward
        for i in range(self.replay_updating_step):
            batch, indices = self.sample(self.replay_updating_batch_size)
            scores = self.scores[indices]
            state, action, param, reward, next_state, done, episode = batch
            temporal_difference = self.td[indices]
            mask  = torch.FloatTensor(np.random.binomial(1, scores)).unsqueeze(1).to(self.device)
            score_features = torch.FloatTensor(np.stack([temporal_difference, current_episode-np.array(episode), reward],axis=1)).to(self.device)
            replay_loss = mask*torch.log(self.score_net(score_features))+(1-mask)*torch.log(1-self.score_net(score_features))
            replay_loss = replay_loss.mean()
            self.score_net_optimizer.zero_grad()
            replay_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), 5)
            self.score_net_optimizer.step()
        subset, indices = self.sample_subset()
        return subset, indices

    def score_update(self, indices, td, current_episode):
        self.td[indices] = td
        batch = self.buffer[indices]
        state, action, param, reward, next_state, done, episode = zip(*batch)
        reward=np.array(reward)
        episode=np.array(episode)
        score_features = torch.FloatTensor(np.stack([td, current_episode-episode, reward],axis=1)).to(self.device)
        scores = self.score_net(score_features).detach().cpu().numpy().squeeze(1)
        if np.any(np.isnan(scores)):
            for name, param in self.score_net.named_parameters():
                if param.requires_grad:
                    print(name, param.data)
            print("NaN score")
        self.scores[indices] = scores


    def sample(self, batch_size):
        indices = np.random.choice(len(self), batch_size)
        batch = self.buffer[indices]
        scores = self.scores[indices]
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst = [], [], [], [], [], [], []
        for sample in batch:
            state, action, last_action, reward, next_state, done, episode = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            episode_lst.append(episode)

        return (s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst), indices
    
    def sample_subset(self):
        mask  = np.random.binomial(1, self.scores)
        indices = mask.nonzero()[0]
        batch = self.buffer[indices]
        scores = self.scores[indices]

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst = [], [], [], [], [], [], []
        for sample in batch:
            state, action, last_action, reward, next_state, done, episode = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            episode_lst.append(episode)
        s_lst=np.array(s_lst)
        a_lst=np.array(a_lst)
        la_lst=np.array(la_lst)
        r_lst=np.array(r_lst)
        ns_lst=np.array(ns_lst)
        d_lst=np.array(d_lst)
        episode_lst=np.array(episode_lst)
        indices=np.array(indices)
        return (s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst), indices

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        return self.buffer[key]

    def compress(self, semantic_length=100):
        compressed_buffer = copy.deepcopy(self)
        for episode in compressed_buffer:
            if episode == 0:
                break
            for i in range(len(episode[0])):
                if i == 0:
                    continue
                episode[0][i] = episode[0][i][:-semantic_length]
        return compressed_buffer

    def uncompress(self, semantic_length=100):
        uncompressed_buffer = copy.deepcopy(self)
        for episode in uncompressed_buffer:
            if episode == 0:
                break
            for i in range(len(episode[0])):
                if i == 0:
                    continue
                episode[0][i] = np.concatenate((episode[0][i], episode[0][0][-semantic_length:]))
        return uncompressed_buffer


Experience = namedtuple('Experience', 'obs1 act last_act rew obs2 done')

class PrioritizedReplayBuffer_SAC2_MLP:
    """ 
    Prioritized Replay Buffer see 
    https://arxiv.org/pdf/1511.05952.pdf and https://cardwing.github.io/files/RL_course_report.pdf for details
    """
    e = 0.01
    d = 100
    alpha = 2.0
    beta = 1.0
    beta_increment_per_sampling = 0.01

    def __init__(self, capacity, reward_l, reward_h, data_bin):
        """ Prioritized experience replay buffer initialization."""
        self.capacity = capacity
        self.buffer = SumTree(int(capacity))
        self._max_priority = 1.0
        self.data_bin = data_bin
        self.bins = []
        self.reward_l = reward_l
        self.reward_h = reward_h
        self.reward_interval = (reward_h-reward_l)/self.data_bin
        for _ in range(self.data_bin):
            self.bins.append(SequenceReplayBuffer(int(capacity/self.data_bin)))

    def _get_priority(self, error):
        """ get priority for TD error"""
        return (np.abs(error) + self.e + self.d) ** self.alpha

    def push(self, state, action, last_action, reward, next_state, done, episode):
        """  push a sample into prioritized replay buffer"""
        self.buffer.add(self._max_priority, (state, action, last_action, reward, next_state, done, episode))
        
        bin_id = int(min(max((reward-self.reward_l)//self.reward_interval,0),self.data_bin-1))
        self.bins[bin_id].push(*copy.deepcopy((state, action, last_action, reward, next_state, done, episode)))

    @property
    def bin_size(self):
        return [len(b) for b in self.bins]
    
    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for idx, error in zip(indices, priorities):
            if idx == -1:
                continue
            p = self._get_priority(error)
            self._max_priority = max(self._max_priority, p)
            self.buffer.update(idx, p)

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """

        batch = []
        indices = []
        weights = []
        segment = self.buffer.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        def clip_small(priority):
            if priority > 1e-16:
                return (1. / self.capacity / priority) ** self.beta
            else:
                return 0

        # sample with strategy
        # make Reward Bin
        data_len = len(self.buffer)
        # reward_sum = [(i, sum(sequence[3])) for i, sequence in enumerate(self.buffer[:data_len])]
        # reward_sum = sorted(reward_sum, key=lambda x: x[1], reverse=True)

        # random sample
        len_random = batch_size - batch_size//2//self.data_bin*self.data_bin
        for i in range(len_random):
            # do not get zero from SumTree
            a = segment * i
            b = segment * (i + 1)
            while True:
                r = random.uniform(a, b)
                index, priority, data = self.buffer.get(r)
                if data:
                    break
            weights.append((1./self.capacity/priority) **
                           self.beta if priority > 1e-16 else 0)
            indices.append(index)
            batch.append(data)  # [batch_size, max_step, 6(state, action_v, param, reward, next_state, done), *]
            # # To avoid duplicating
            # self.buffer.update(index, 0)

        avg_weight = np.mean(weights)
        i=-1
        while len(batch)<batch_size:
            i=(i+1)%self.data_bin
            if len(self.bins[i])==0: 
                continue
            data = self.bins[i].sample(batch_size//2//self.data_bin)
            i=(i+1)%self.data_bin
            # append to list
            for d in data:
                weights.append(avg_weight)
                indices.append(-1)
                batch.append(d)
            # for every bin, fetch batch_size//(2 * self.data_bin) data items

        # sampling_priorities = priorities / self.buffer.total()
        # sampling_priorities = sorted(sampling_priorities, reverse=True)
        # print("[Debug] First 20 Sample Priorities:", sampling_priorities[:20])

        # Revert priorities
        # for idx, p in zip(indices, priorities):
        #     self.buffer.update(idx, p)

        # Normalize for stability
        weights = np.array(weights)
        #weights /= max(weights)

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst = [], [], [], [], [], [], []
        for sample in batch:
            state, action, last_action, reward, next_state, done, episode = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            episode_lst.append(episode)

        return (s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst), indices, weights

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer.data[key]

    def compress(self, semantic_length=100):
        compressed_buffer = copy.deepcopy(self)
        for episode in compressed_buffer:
            if episode == 0:
                break
            for i in range(len(episode[0])):
                if i == 0:
                    continue
                episode[0][i] = episode[0][i][:-semantic_length]
        for abin in compressed_buffer.bins:
            for episode in abin:
                if episode == 0:
                    break
                for i in range(len(episode[0])):
                    if i == 0:
                        continue
                    episode[0][i] = episode[0][i][:-semantic_length]
        return compressed_buffer

    def uncompress(self, semantic_length=100):
        uncompressed_buffer = copy.deepcopy(self)
        for episode in uncompressed_buffer:
            if episode == 0:
                break
            for i in range(len(episode[0])):
                if i == 0:
                    continue
                episode[0][i] = np.concatenate((episode[0][i], episode[0][0][-semantic_length:]))
        for abin in uncompressed_buffer.bins:
            for episode in abin:
                if episode == 0:
                    break
                for i in range(len(episode[0])):
                    if i == 0:
                        continue
                    episode[0][i] = np.concatenate((episode[0][i], episode[0][0][-semantic_length:]))
        return uncompressed_buffer


class PrioritizedReplayBuffer_SAC3_MLP:
    """ 
    Prioritized Replay Buffer see 
    https://arxiv.org/pdf/1511.05952.pdf and https://cardwing.github.io/files/RL_course_report.pdf for details
    """

    def __init__(self, capacity, capacity_distribution, reward_l, reward_h, data_bin):
        """ Prioritized experience replay buffer initialization."""
        self.capacity = capacity
        self._max_priority = 1.0
        self.bins = []
        self.data_bin = data_bin
        self.capacity_distribution=capacity_distribution
        self.reward_l = reward_l
        self.reward_h = reward_h
        self.interval = (self.reward_h-self.reward_l)/self.data_bin
        if capacity_distribution=='uniform':
            for _ in range(self.data_bin):
                self.bins.append(PrioritizedReplayBuffer_Original(capacity//self.data_bin))
        elif capacity_distribution=='exponential':
            for i in range(self.data_bin):
                self.bins.append(PrioritizedReplayBuffer_Original(capacity//(2**(self.data_bin-i))))

    def push(self, state, action, last_action, reward, next_state, done, episode):
        """  push a sample into prioritized replay buffer"""
        bin_id = int(min(max((reward-self.reward_l)//self.interval,0),self.data_bin-1))
        self.bins[bin_id].push(state, action, last_action, reward, next_state, done, episode)
    
    @property
    def bin_size(self):
        return [len(b) for b in self.bins]
    
    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for idx, error in zip(indices, priorities):
            if idx == -1:
                continue
            if self.capacity_distribution=='uniform':
                bin_id = idx//(self.capacity//self.data_bin*2-1)
            elif self.capacity_distribution == 'exponential':
                bin_id = 0
                capacity_sum = 0
                while capacity_sum+self.bins[bin_id].buffer.tree_size<=idx:
                    capacity_sum+=self.bins[bin_id].buffer.tree_size
                    bin_id+=1
            idx_in_bin = idx - sum([abin.buffer.tree_size for abin in self.bins[:bin_id]])
            if idx_in_bin<0:
                print(idx_in_bin)
            self.bins[bin_id].priority_update([idx_in_bin],[error])

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """

        assert isinstance(batch_size, int) or isinstance(batch_size, np.int64) or len(batch_size)==self.data_bin, "Batch size must be a number or a list as long as the list of bins"
        indices = []
        weights = []
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst = [], [], [], [], [], [], []

        if isinstance(batch_size, int) or isinstance(batch_size, np.int64):
            i = -1
            while batch_size>0:
                i=(i+1)%self.data_bin
                if len(self.bins[i])==0: 
                    continue
                _batch, _indices, _weights = self.bins[i].sample(1)
                _indices = (np.array(_indices) + sum([abin.buffer.tree_size for abin in self.bins[:i]])).tolist()
                _s_lst, _a_lst, _la_lst, _r_lst, _ns_lst, _d_lst, _episode_lst = _batch
                s_lst = s_lst + _s_lst
                a_lst = a_lst + _a_lst
                la_lst = la_lst + _la_lst
                r_lst = r_lst + _r_lst
                ns_lst = ns_lst + _ns_lst
                d_lst = d_lst + _d_lst
                indices = indices + _indices
                weights.append(_weights)
                episode_lst = episode_lst + _episode_lst
                batch_size -= 1
        else:
            for i in range(self.data_bin):
                _batch, _indices, _weights = self.bins[i].sample(batch_size[i])
                _indices = (np.array(_indices) + sum([abin.buffer.tree_size for abin in self.bins[:i]])).tolist()
                _s_lst, _a_lst, _la_lst, _r_lst, _ns_lst, _d_lst, _episode_lst = _batch
                s_lst = s_lst + _s_lst
                a_lst = a_lst + _a_lst
                la_lst = la_lst + _la_lst
                r_lst = r_lst + _r_lst
                ns_lst = ns_lst + _ns_lst
                d_lst = d_lst + _d_lst
                indices = indices + _indices
                weights.append(_weights)
                episode_lst = episode_lst + _episode_lst
        weights = np.concatenate(weights)
        return (s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst), indices, weights

    def __len__(self):
        return sum([len(_bin.buffer) for _bin in self.bins])

    def compress(self, semantic_length=100):
        compressed_buffer = copy.deepcopy(self)
        for abin in compressed_buffer.bins:
            for episode in abin:
                if episode == 0:
                    break
                for i in range(len(episode[0])):
                    if i == 0:
                        continue
                    episode[0][i] = episode[0][i][:-semantic_length]
        return compressed_buffer

    def uncompress(self, semantic_length=100):
        uncompressed_buffer = copy.deepcopy(self)
        for abin in uncompressed_buffer.bins:
            for episode in abin:
                if episode == 0:
                    break
                for i in range(len(episode[0])):
                    if i == 0:
                        continue
                    episode[0][i] = np.concatenate((episode[0][i], episode[0][0][-semantic_length:]))
        return uncompressed_buffer
class NormalizedHybridActions(gym.ActionWrapper):
    """ Action Normalization: just normalize 2nd element in action tuple (id, (params)) """

    def action(self, action: tuple) -> tuple:
        low = self.action_space[1].low
        high = self.action_space[1].high

        param = 0.5 * (action[1] + 1.0) * (high - low) + low
        param = np.clip(param, low, high)

        return action[0], param

    def reverse_action(self, action: tuple) -> tuple:
        low = self.action_space[1].low
        high = self.action_space[1].high
        param = np.clip(action[1], low, high)
        param = 2.0 * (param - low) / (high - low) - 1

        return action[0], param


class OUNoise(object):
    """
    Exploration Noise for deterministic policy methods
    https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    """

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action: np.ndarray, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - \
            (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return (action + ou_state).clip(self.low, self.high)


class GaussianNoise(object):
    """
    This strategy adds Gaussian noise to the action taken by the deterministic policy.
    https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/gaussian_strategy.py
    """

    def __init__(self, action_space, max_sigma=1.0, min_sigma=None, decay_period=1000000):
        assert len(action_space.shape) == 1
        self._max_sigma = max_sigma
        if min_sigma is None:
            min_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self._action_space = action_space

    def get_action(self, action, t=None):
        sigma = (
            self._max_sigma - (self._max_sigma - self._min_sigma) *
            min(1.0, t * 1.0 / self._decay_period)
        )
        return np.clip(
            action + np.random.normal(size=len(action)) * sigma,
            self._action_space.low,
            self._action_space.high,
        )


class GaussianAndEpislonNoise(object):
    """
    With probability epsilon, take a completely random action.
    with probability 1-epsilon, add Gaussian noise to the action taken by a
    deterministic policy.
    https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/gaussian_and_epsilon_strategy.py
    """

    def __init__(self, action_space, epsilon, max_sigma=1.0, min_sigma=None,
                 decay_period=1000000):
        assert len(action_space.shape) == 1
        if min_sigma is None:
            min_sigma = max_sigma
        self._max_sigma = max_sigma
        self._epsilon = epsilon
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self._action_space = action_space

    def get_action(self, action, t=None, **kwargs):
        if random.random() < self._epsilon:
            return self._action_space.sample()
        else:
            sigma = self._max_sigma - \
                (self._max_sigma - self._min_sigma) * \
                min(1.0, t * 1.0 / self._decay_period)
            return np.clip(
                action + np.random.normal(size=len(action)) * sigma,
                self._action_space.low,
                self._action_space.high,
            )


class ParamNoise(object):
    def __init__(self, param_std, action_std, adapt_coefficient):
        """
        Parameter Noise for actor networks
        :param param_std: init variance for parameter noise
        :param action_std: desired action variance (to estimate policy distance)
        :param adapt_coefficient: coefficient of adaptation of parameter noise each time
        """
        self.param_std, self.action_std, self.adapt_coefficient = param_std, action_std, adapt_coefficient
        # deep copy before each episode & perturb
        self.net, self.perturb_net = None, None

    @staticmethod
    def _get_distance(perturbed_actions: tuple, original_actions: tuple, weights=(1., 1.)):
        """
        approximate distance of two policies (action batches) with discrete and continuous parts (tuple of tensors)
        (delta = -log(1 - epsilon + epsilon / |A|) for DQN & = delta for DDPG)
        :param weights: weights for difference between discrete and continuous action parts
        """
        return sum([((original_actions[i] - perturbed_actions[i]).pow(2).mean(dim=0).mean().pow(0.5)) * weights[i]
                    for i in range(2)])

    def _adapt(self, distance):
        """ adapt the parameter noise variance to a proper value according to the perturbed action distance """
        # If output action's L2 distance bewteen perturbed net and unperturbed net > threshold
        # std *= coef, otherwise /= coef
        self.param_std = (self.param_std / self.adapt_coefficient if distance > self.action_std
                          else self.param_std * self.adapt_coefficient)

    def perturb(self, net: nn.Module, param_std=None, is_init=True, need_print=False):
        """ perturb network parameters with Gaussian noise (at the beginning of each episode) """
        if param_std is not None:
            self.param_std = param_std
        if is_init and need_print:
            print('[debug] perturb std:', self.param_std)
        with torch.no_grad():
            self.net = net
            self.perturb_net = copy.deepcopy(net)
            self.perturb_net.eval()
            for fc in self.perturb_net.get_fc():
                for param in fc.parameters():
                    # create new same size and device rand tensor
                    param_inc = torch.randn_like(param) * self.param_std
                    param += param_inc
            # for name, param in net.named_parameters():
            #     print(name, '  ', param)
            # for name, param in self.perturb_net.named_parameters():
            #     print(name, '  ', param)

    def get_action(self, state, hidden_state, cell_state, evaluate=True, original_actions=None):
        """ get perturbed actions (at each step / update) """
        if evaluate and original_actions is None:  # not update if evaluating (e.g. target action exploration in TD3)
            action_enc, action_v, param, hidden_state_, cell_state_ = self.perturb_net.get_action(
                state, hidden_state, cell_state)
            # bug fixed: hidden_state and cell state cannot be iterated
            # (action_v_, param_) = action_enc
            with torch.no_grad():
                self._adapt(self._get_distance(action_enc,
                            self.net.get_action(state, hidden_state, cell_state)[0]
                            if original_actions is None else original_actions))
        else:
            action_v, param, (hidden_state_, cell_state_) = self.perturb_net(
                state, hidden_state, cell_state)
            self._adapt(self._get_distance((action_v, param),
                        self.net(state, hidden_state, cell_state)[
                :-1]
                if original_actions is None else original_actions))
            self.perturb(self.net, param_std=self.param_std)
        return action_v, param, hidden_state_, cell_state_


if __name__ == '__main__':
    # test param noise
    param_noise = ParamNoise(
        param_std=0.05, action_std=0.3, adapt_coefficient=1.01)
    from Utils_RL.models import HybridPolicyNetwork
    net = HybridPolicyNetwork(10, 3, 3, 128).cuda()

    param_noise.perturb(net)
    print(param_noise.get_action(np.ones(10), evaluate=True))
    param_noise.perturb(net)
    print(param_noise.get_action(np.ones(10), evaluate=True))
    param_noise.perturb(net)
    print(param_noise.get_action(np.ones(10), evaluate=True))

def is_prioritized(buffer):
    return isinstance(buffer, PrioritizedReplayBuffer_SAC) or \
        isinstance(buffer, PrioritizedReplayBuffer_SAC2) or \
        isinstance(buffer, PrioritizedReplayBuffer_SAC3) or \
        isinstance(buffer, PrioritizedReplayBuffer_Original)

def has_multiple_bins(buffer):
    return isinstance(buffer, PrioritizedReplayBuffer_SAC2) or isinstance(buffer, PrioritizedReplayBuffer_SAC3)