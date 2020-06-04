import torch
from torch.nn import functional as F
import numpy as np
import copy
import datetime
import csv
import codecs
import gym
import os
import warnings
from torch.utils.tensorboard import SummaryWriter

from Experiments_RL.sac_lstm import PASAC_Agent_LSTM
from Experiments_RL.sac_mlp import PASAC_Agent_MLP
from Experiments_RL.sac_dem_mlp import PASAC_Agent_DEM_MLP
from Experiments_RL.sac_ero import PASAC_Agent_ERO

from Utils_RL.utils import PrioritizedReplayBuffer_LSTM, PrioritizedReplayBuffer_MLP

from Benchmarks.utils_benchmark import ActionUnwrap, StateUnwrap
from Benchmarks.common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper, \
    QPAMDPScaledParameterisedActionWrapper
from Benchmarks.common.platform_domain import PlatformFlattenedActionWrapper
from Utils_RL.utils import NormalizedHybridActions, v2id, PATransition
import matplotlib.pyplot as plt

import logging
#import nni

logger = logging.getLogger('benchmark_goal_tuning')


def train_nni(**kwargs):
    params = locals()['kwargs']  # get local parameters (in dict kwargs), including all arguments
    if params['use_nni']:
        try:
            # get parameters form tuner
            tuner_params = nni.get_next_parameter()
            print(params, tuner_params)
            params.update(tuner_params)
        except Exception as exception:
            logger.exception(exception)
            raise
        print('[params after NNI]', params)
    train_mlp(**params)


def train(env_name, debug,
          seed, max_steps, train_episodes,
          batch_size, update_freq, eval_freq,
          weights, gamma, replay_buffer_size,
          hidden_size, value_lr, policy_lr, soft_tau=1e-2,
          use_exp=True, use_nni=False, rnn_step=10):
    assert env_name in ['Platform-v0', 'Goal-v0']
    if 'Goal' in env_name:
        import gym_goal
    if 'Platform' in env_name:
        import gym_platform
    env = gym.make(env_name)
    env = ScaledStateWrapper(env)

    env = ActionUnwrap(env)  #  scale to [-1,1] to match the range of tanh
    env = StateUnwrap(env)
    env = NormalizedHybridActions(env)

    # env specific
    state_dim = env.observation_space.shape[0]
    action_discrete_dim, action_continuous_dim = env.action_space.spaces[0].n, env.action_space.spaces[1].shape[0]

    env.seed(seed)
    np.random.seed(seed)

    agent = PASAC_Agent_LSTM(debug, weights, gamma, replay_buffer_size, rnn_step,
                             hidden_size, value_lr, policy_lr, batch_size, state_dim,
                             action_discrete_dim, action_continuous_dim, soft_tau,
                             use_exp)

    if isinstance(agent.replay_buffer, PrioritizedReplayBuffer_LSTM):
        agent.replay_buffer.beta_increment_per_sampling = 1. / (max_steps * train_episodes)
    returns = []
    warnings.filterwarnings("ignore")
    start = datetime.datetime.now()
    DIR = debug['tensorboard_dir']
    NAME = os.path.join(DIR, str(start.strftime('%m.%d-%H-%M-%S') + env_name))
    print(NAME)
    if not use_nni:
        writer = SummaryWriter(NAME)

    for episode in range(train_episodes):
        # -----save model-----
        if debug['save_model'] and episode % debug['save_freq'] == 0 and episode > 0:
            print('============================================')
            print("Savepoint - Save model in episodes:", episode)
            print('============================================')
            agent.save(episode)

        # -----reset env-----
        state = env.reset()

        # -----init-----
        episode_reward_sum = 0.
        last_action = (np.zeros(action_discrete_dim), np.zeros(action_continuous_dim))
        episode_state = []
        episode_action_v = []
        episode_param = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []
        episode_done = []
        hidden_out = agent.policy_net.init_hidden_states(bsize=1)

        for step in range(max_steps):
            # -----step-----
            agent.total_step += 1
            hidden_in = hidden_out
            action, _, action_v, param, hidden_out = agent.act(state, hidden_in, debug['sampling'])
            next_state, reward, done, _ = env.step(action)

            # -----append step to the sequence-----
            episode_state.append(state)
            episode_action_v.append(action_v)
            episode_param.append(param)
            episode_last_action.append(last_action)
            episode_reward.append(reward)
            episode_next_state.append(next_state)
            episode_done.append(done)

            # -----move to the next step-----
            state = next_state
            last_action = action
            episode_reward_sum += reward

            # -----update models-----
            if len(agent.replay_buffer) > batch_size and step % update_freq == 0:
                agent.update(batch_size,
                             auto_entropy=True,
                             soft_tau=soft_tau,
                             target_entropy=-1. * (action_continuous_dim),
                             need_print=(episode % debug['print_freq'] == 0) and step == 0)

            # -----done-----
            if done:
                break

        # -----add seq to replay buffer-----
        agent.replay_buffer.push(episode_state, episode_action_v, episode_param,
                                 episode_reward, episode_next_state, episode_done)

        if episode % 100 == 0:
            print(f'episode: {episode}, reward: {episode_reward_sum}')
        returns.append(episode_reward_sum)
        if not use_nni:
            writer.add_scalar('Training-Reward-' + env_name, episode_reward_sum, global_step=episode)

        # [periodic evaluation]
        if episode % 10 == 0:  # more frequent
            print(episode, '[time]', datetime.datetime.now() - start,
                  episode_reward_sum, '\n', '>>>>>>>>>>>>>>>>')

            # [evaluation]
            episode_reward_eval = evaluate(agent, env, max_steps, use_nni, eval_repeat=1)
            if not use_nni:
                writer.add_scalar('EvalReward-' + env_name, episode_reward_eval, global_step=episode)

    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    # [report final results]
    average_reward = sum(returns) / len(returns)
    evaluate(agent, env, max_steps, use_nni, report_avg=average_reward, eval_repeat=100)  # less time

    env.close()
    if not use_nni:
        writer.close()


def evaluate(agent, env, max_steps, use_nni=False, report_avg=None, eval_repeat=1):
    print("Evaluating agent over {} episodes".format(eval_repeat))
    evaluation_returns = []
    hidden_out = agent.policy_net.init_hidden_states(bsize=1)
    for _ in range(eval_repeat):
        state = env.reset()
        episode_reward = 0.
        for _ in range(max_steps):
            with torch.no_grad():
                hidden_in = hidden_out
                action, _, _, _, hidden_out = agent.act(state, hidden_in, True)
                next_state, reward, done, _ = env.step(action)

                state = next_state
                episode_reward += reward
            if done:  # currently all situations end with a done
                break

        evaluation_returns.append(episode_reward)
    eval_avg = sum(evaluation_returns) / len(evaluation_returns)
    print("Ave. evaluation return =", eval_avg)

    if use_nni:
        if eval_repeat == 1:
            nni.report_intermediate_result(eval_avg)
        elif eval_repeat > 1 and report_avg is not None:
            metric = (report_avg + eval_avg) / 2
            nni.report_final_result(metric)
    return eval_avg


def train_mlp(env_name, debug,
              seed, max_steps, train_episodes,
              batch_size, update_freq, eval_freq,
              weights, gamma, replay_buffer_size,
              hidden_size, value_lr, policy_lr,
              soft_tau=1e-2,
              use_exp=True,
              use_nni=False):
    assert env_name in ['Platform-v0', 'Goal-v0']
    if 'Goal' in env_name:
        import gym_goal
    if 'Platform' in env_name:
        import gym_platform
    env = gym.make(env_name)
    env = ScaledStateWrapper(env)

    env = ActionUnwrap(env)  #  scale to [-1,1] to match the range of tanh
    env = StateUnwrap(env)
    env = NormalizedHybridActions(env)

    # env specific
    state_dim = env.observation_space.shape[0]
    action_discrete_dim, action_continuous_dim = env.action_space.spaces[0].n, env.action_space.spaces[1].shape[0]

    env.seed(seed)
    np.random.seed(seed)

    agent = PASAC_Agent_MLP(debug, weights, gamma, replay_buffer_size, max_steps,
                            hidden_size, value_lr, policy_lr, batch_size, state_dim,
                            action_discrete_dim, action_continuous_dim, soft_tau,
                            use_exp)

    if isinstance(agent.replay_buffer, PrioritizedReplayBuffer_MLP):
        agent.replay_buffer.beta_increment_per_sampling = 1. / (max_steps * train_episodes)
    returns = []
    warnings.filterwarnings("ignore")
    start = datetime.datetime.now()
    DIR = debug['tensorboard_dir']
    NAME = os.path.join(DIR, str(start.strftime('%m.%d-%H-%M-%S') + env_name))
    print(NAME)
    if not use_nni:
        writer = SummaryWriter(NAME)

    for episode in range(train_episodes):
        # -----save model-----
        if debug['save_model'] and episode % debug['save_freq'] == 0 and episode > 0:
            print('============================================')
            print("Savepoint - Save model in episodes:", episode)
            print('============================================')
            agent.save(episode)

        # -----reset env-----
        state = env.reset()

        # -----init-----
        episode_reward_sum = 0.

        for step in range(max_steps):
            # -----step-----
            agent.total_step += 1
            action, _, action_v, param = agent.act(state, debug['sampling'])
            next_state, reward, done, _ = env.step(action)

            agent.replay_buffer.push(state, action_v, param, reward, next_state, done)

            # -----move to the next step-----
            state = next_state
            episode_reward_sum += reward

            # -----update models-----
            if len(agent.replay_buffer) > batch_size and step % update_freq == 0:
                agent.update(batch_size,
                             auto_entropy=True,
                             soft_tau=soft_tau,
                             target_entropy=-1. * (action_continuous_dim),
                             need_print=(episode % debug['print_freq'] == 0) and step == 0)

            # -----done-----
            if done:
                break

        if episode % 100 == 0:
            print(f'episode: {episode}, reward: {episode_reward_sum}')
        returns.append(episode_reward_sum)
        if not use_nni:
            writer.add_scalar('Training-Reward-' + env_name, episode_reward_sum, global_step=episode)

        # [periodic evaluation]
        if episode % 10 == 0:  # more frequent
            print(episode, '[time]', datetime.datetime.now() - start,
                  episode_reward_sum, '\n', '>>>>>>>>>>>>>>>>')

            # [evaluation]
            episode_reward_eval = evaluate_mlp(agent, env, max_steps, use_nni, eval_repeat=1)
            if not use_nni:
                writer.add_scalar('EvalReward-' + env_name, episode_reward_eval, global_step=episode)

    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    # [report final results]
    average_reward = sum(returns) / len(returns)
    evaluate_mlp(agent, env, max_steps, use_nni, report_avg=average_reward, eval_repeat=100)  # less time

    env.close()
    if not use_nni:
        writer.close()


def evaluate_mlp(agent, env, max_steps, use_nni=False, report_avg=None, eval_repeat=1):
    print("Evaluating agent over {} episodes".format(eval_repeat))
    evaluation_returns = []
    for _ in range(eval_repeat):
        state = env.reset()
        episode_reward = 0.
        for _ in range(max_steps):
            with torch.no_grad():
                action, _, _, _ = agent.act(state, True)
                next_state, reward, done, _ = env.step(action)

                state = next_state
                episode_reward += reward
            if done:  # currently all situations end with a done
                break

        evaluation_returns.append(episode_reward)
    eval_avg = sum(evaluation_returns) / len(evaluation_returns)
    print("Ave. evaluation return =", eval_avg)

    if use_nni:
        if eval_repeat == 1:
            nni.report_intermediate_result(eval_avg)
        elif eval_repeat > 1 and report_avg is not None:
            metric = (report_avg + eval_avg) / 2
            nni.report_final_result(metric)
    return eval_avg


def train_dem_mlp(log_name, env_name, debug,
          seed, max_steps, train_episodes,
          batch_size, update_freq, eval_freq,
          weights, gamma, replay_buffer_size,
          hidden_size, value_lr, policy_lr, soft_tau=1e-2,
          use_exp=True, use_nni=False, rnn_step=10, discrete_log_prob_scale=10):
    assert env_name in ['Platform-v0', 'Goal-v0']
    if 'Goal' in env_name:
        import gym_goal
        reward_l = -15
        reward_h = 15
        data_bin = 3
    if 'Platform' in env_name:
        import gym_platform
        reward_l = 0
        reward_h = 0.15
        data_bin = 4
    env = gym.make(env_name)
    env = ScaledStateWrapper(env)

    env = ActionUnwrap(env)  #  scale to [-1,1] to match the range of tanh
    env = StateUnwrap(env)
    env = NormalizedHybridActions(env)

    # env specific
    state_dim = env.observation_space.shape[0]
    action_discrete_dim, action_continuous_dim = env.action_space.spaces[0].n, env.action_space.spaces[1].shape[0]

    env.seed(seed)
    np.random.seed(seed)

    agent = PASAC_Agent_DEM_MLP(env, debug, weights, gamma, replay_buffer_size, max_steps,
                             hidden_size, value_lr, policy_lr, batch_size, state_dim,
                             action_discrete_dim, action_continuous_dim, soft_tau,
                             use_exp, discrete_log_prob_scale, reward_l, reward_h, data_bin)

    if isinstance(agent.replay_buffer, PrioritizedReplayBuffer_LSTM):
        agent.replay_buffer.beta_increment_per_sampling = 1. / (max_steps * train_episodes)
    returns = []
    warnings.filterwarnings("ignore")
    start = datetime.datetime.now()
    DIR = debug['tensorboard_dir']
    NAME = os.path.join(DIR, str(start.strftime('%m.%d-%H-%M-%S') + log_name))
    print(NAME)
    if not use_nni:
        writer = SummaryWriter(NAME)

    for episode in range(train_episodes):
        # -----save model-----
        if debug['save_model'] and episode % debug['save_freq'] == 0 and episode > 0:
            print('============================================')
            print("Savepoint - Save model in episodes:", episode)
            print('============================================')
            agent.save(episode)

        # -----reset env-----
        state = env.reset()

        # -----init-----
        episode_reward_sum = 0.

        for step in range(max_steps):
            # -----step-----
            agent.total_step += 1
            action, _, action_v, param = agent.act(state, debug['sampling'])
            next_state, reward, done, _ = env.step(action)

            agent.replay_buffer.push(state, action_v, param, reward, next_state, done, episode)

            # -----move to the next step-----
            state = next_state
            episode_reward_sum += reward

            # -----update models-----
            if len(agent.replay_buffer) > batch_size and step % update_freq == 0:
                agent.update(batch_size,
                             episode=episode,
                             auto_entropy=True,
                             soft_tau=soft_tau,
                             target_entropy=-1. * (action_continuous_dim),
                             need_print=(episode % debug['print_freq'] == 0) and step == 0)

            # -----done-----
            if done:
                break        

        if episode % 100 == 0:
            print(f'episode: {episode}, reward: {episode_reward_sum}')
        returns.append(episode_reward_sum)
        if not use_nni:
            writer.add_scalar('Training-Reward-' + env_name, episode_reward_sum, global_step=episode)

        # [periodic evaluation]
        if episode % 10 == 0:  # more frequent
            print(episode, '[time]', datetime.datetime.now() - start,
                  episode_reward_sum, '\n', '>>>>>>>>>>>>>>>>')

            # [evaluation]
            episode_reward_eval = evaluate_mlp(agent, env, max_steps, use_nni, eval_repeat=10)
            if not use_nni:
                writer.add_scalar('EvalReward-' + env_name, episode_reward_eval, global_step=episode)
        
        if episode % 10000 < 3:
            # print("Replay buffer in episode " + str(episode))
            # agent.replay_buffer.print_status()
            _,_,_,reward_arr,_,_,_,_,_ = zip(*agent.replay_buffer.buffer[:len(agent.replay_buffer)])
            id_arr = np.arange(len(agent.replay_buffer))
            priority_arr = agent.replay_buffer.priorities
            plt.scatter(x=reward_arr, y=priority_arr, marker='.', s=1)
            plt.savefig('plot/'+start.strftime('%m.%d-%H-%M-%S') + env_name + str(episode) + 'rw_pr.pdf')  
            plt.clf()
            plt.scatter(x=id_arr, y=priority_arr, marker='.', s=1)
            plt.savefig('plot/'+start.strftime('%m.%d-%H-%M-%S') + env_name + str(episode) + 'id_pr.pdf')
            plt.clf()
            

    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    # [report final results]
    average_reward = sum(returns) / len(returns)
    evaluate_mlp(agent, env, max_steps, use_nni, report_avg=average_reward, eval_repeat=100)  # less time

    env.close()
    if not use_nni:
        writer.close()

def train_ero(log_name, env_name, debug,
          seed, max_steps, train_episodes,
          batch_size, update_freq, eval_freq,
          weights, gamma, replay_buffer_size,
          hidden_size, value_lr, policy_lr, soft_tau=1e-2,
          use_exp=True, use_nni=False, rnn_step=10, discrete_log_prob_scale=10):
    training_step = 50
    assert env_name in ['Platform-v0', 'Goal-v0']
    if 'Goal' in env_name:
        import gym_goal
        reward_l = -15
        reward_h = 15
        data_bin = 3
    if 'Platform' in env_name:
        import gym_platform
        reward_l = 0
        reward_h = 0.15
        data_bin = 4
    env = gym.make(env_name)
    env = ScaledStateWrapper(env)

    env = ActionUnwrap(env)  #  scale to [-1,1] to match the range of tanh
    env = StateUnwrap(env)
    env = NormalizedHybridActions(env)

    env1 = gym.make(env_name)
    env1 = ScaledStateWrapper(env1)

    env1 = ActionUnwrap(env1)  #  scale to [-1,1] to match the range of tanh
    env1 = StateUnwrap(env1)
    env1 = NormalizedHybridActions(env1)

    # env specific
    state_dim = env.observation_space.shape[0]
    action_discrete_dim, action_continuous_dim = env.action_space.spaces[0].n, env.action_space.spaces[1].shape[0]

    env.seed(seed)
    np.random.seed(seed)

    agent = PASAC_Agent_ERO(env, debug, weights, gamma, replay_buffer_size, max_steps,
                             hidden_size, value_lr, policy_lr, batch_size, state_dim,
                             action_discrete_dim, action_continuous_dim, soft_tau,
                             use_exp, discrete_log_prob_scale, reward_l, reward_h, data_bin)

    returns = []
    warnings.filterwarnings("ignore")
    start = datetime.datetime.now()
    DIR = debug['tensorboard_dir']
    NAME = os.path.join(DIR, str(start.strftime('%m.%d-%H-%M-%S') + log_name))
    print(NAME)
    if not use_nni:
        writer = SummaryWriter(NAME)
    previous_cumulative_reward = 0.
    state0 = env.reset()
    action0, _, action_v0, param0 = agent.act(state0, debug['sampling'])
    for episode in range(train_episodes):
        # -----save model-----
        if debug['save_model'] and episode % debug['save_freq'] == 0 and episode > 0:
            print('============================================')
            print("Savepoint - Save model in episodes:", episode)
            print('============================================')
            agent.save(episode)

        # -----reset env-----
        state = env.reset()
        next_state0, reward0, done0, _ = env.step(action0)

        # -----init-----
        episode_reward_sum = 0.

        for step in range(max_steps):
            # -----step-----
            agent.total_step += 1
            action, _, action_v, param = agent.act(state, debug['sampling'])
            next_state, reward, done, _ = env.step(action)
            state_=torch.FloatTensor([state]).to(agent.device)
            action_v_=torch.FloatTensor([action_v]).to(agent.device)
            param_=torch.FloatTensor([param]).to(agent.device)
            next_state_=torch.FloatTensor([next_state]).to(agent.device)
            predicted_q_value1 = agent.soft_q_net1(state_, action_v_, param_)
            predicted_q_value2 = agent.soft_q_net2(state_, action_v_, param_)
            new_action_v, new_param, log_prob, _, _, _ = agent.policy_net.evaluate(state_)
            new_next_action_v, new_next_param, next_log_prob, _, _, _ = agent.policy_net.evaluate(next_state_)
            predict_q1 = agent.soft_q_net1(state_, new_action_v, new_param)
            predict_q2 = agent.soft_q_net2(state_, new_action_v, new_param)
            predicted_new_q_value = torch.min(predict_q1, predict_q2)
            predict_target_q1 = agent.target_soft_q_net1(next_state_, new_next_action_v, new_next_param)
            predict_target_q2 = agent.target_soft_q_net2(next_state_, new_next_action_v, new_next_param)
            action_sum_log_prob = agent.get_log_action_prob(new_action_v, agent.use_exp)
            action_sum_log_prob_next = agent.get_log_action_prob(new_next_action_v, agent.use_exp)
            policy_loss = agent.alpha_c * log_prob + agent.alpha_d * action_sum_log_prob - predicted_new_q_value
            target_q_min = torch.min(predict_target_q1, predict_target_q2) - agent.alpha_c * next_log_prob - agent.alpha_d * action_sum_log_prob_next
            target_q_value = reward + (1 - done) * gamma * target_q_min
            q_value_loss1 = predicted_q_value1 - target_q_value.detach()
            q_value_loss2 = predicted_q_value2 - target_q_value.detach()
            error = (q_value_loss1**2+q_value_loss2**2+policy_loss**2*agent.w_policy_loss).detach().cpu().numpy().squeeze()

            agent.replay_buffer.push(state, action_v, param, reward, next_state, done, episode, error)

            # -----move to the next step-----
            state = next_state

            # -----done-----
            if done:
                break
        env1.seed(0)
        state = env1.reset()
        for step in range(max_steps):
            # -----step-----
            action, _, action_v, param = agent.act(state, debug['sampling'])
            next_state, reward, done, _ = env1.step(action)

            # -----move to the next step-----
            state = next_state
            episode_reward_sum += reward

            # -----done-----
            if done:
                break
        avg_replay_loss = agent.update_replay_policy(episode_reward_sum, previous_cumulative_reward, episode)
        writer.add_scalar('Replay-Loss-' + env_name, avg_replay_loss, global_step=episode)
        previous_cumulative_reward = episode_reward_sum

        # -----update models-----
        avg_total_error_lst = []
        for _ in range(training_step):
            if len(agent.replay_buffer) > batch_size and step % update_freq == 0:
                avg_predicted_new_q_value, avg_total_error = agent.update(batch_size,
                             episode=episode,
                             auto_entropy=True,
                             soft_tau=soft_tau,
                             target_entropy=-1. * (action_continuous_dim),
                             need_print=(episode % debug['print_freq'] == 0) and step == 0)
                avg_total_error_lst.append(avg_total_error)
        writer.add_scalar('Total-Error-' + env_name, np.asarray(avg_total_error_lst).mean(), global_step=episode)

        if episode % 100 == 0:
            print(f'episode: {episode}, reward: {episode_reward_sum}')
            for name, param in agent.replay_buffer.score_net.named_parameters():
                if param.requires_grad:
                    print(name, param.data)
        returns.append(episode_reward_sum)
        if not use_nni:
            writer.add_scalar('Training-Reward-' + env_name, episode_reward_sum, global_step=episode)

        # [periodic evaluation]
        if episode % 10 == 0:  # more frequent
            print(episode, '[time]', datetime.datetime.now() - start,
                  episode_reward_sum, '\n', '>>>>>>>>>>>>>>>>')

            # [evaluation]
            episode_reward_eval = evaluate_mlp(agent, env, max_steps, use_nni, eval_repeat=1)
            if not use_nni:
                writer.add_scalar('EvalReward-' + env_name, episode_reward_eval, global_step=episode)
        
        # if episode % 10000 < 3:
        #     print("Replay buffer in episode " + str(episode))
        #     for i in range(len(agent.replay_buffer)):
        #         experience = agent.replay_buffer[i]
        #         print('rw: ', str(experience.rew), 'td: ', str(experience.td), 'ep: ', str(experience.episode), 'st: ', str(experience.st))

    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    # [report final results]
    average_reward = sum(returns) / len(returns)
    evaluate(agent, env, max_steps, use_nni, report_avg=average_reward, eval_repeat=100)  # less time

    env.close()
    if not use_nni:
        writer.close()
