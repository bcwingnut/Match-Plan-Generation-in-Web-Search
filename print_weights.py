import click
from Utils_RL.models import *


@click.command()
@click.option('--env', default='Platform-v0')  # = Platform-v0 / Goal-v0
@click.option('--seed', default=0)
@click.option('--log_name', default="Platform_ero")
@click.option('--weights', default=[1., -1, 0.])  # [reward design]
@click.option('--gamma', default=0.95)
@click.option('--replay_buffer_size', default=50000)
# [network]
@click.option('--hidden_size', default=128)
@click.option('--value_lr', default=0.0023631049207951177)
@click.option('--policy_lr', default=0.0004617289434609426)
# [training]
@click.option('--train_episodes', default=500000)
@click.option('--batch_size', default=128)
@click.option('--update_freq', default=1)
@click.option('--eval_freq', default=10)
@click.option('--use_exp', default=True)  # True = exp, False = Sample
@click.option('--soft_tau', default=0.01884329222019157)
# [test param]
@click.option('--tensorboard_dir', default="./tensorboard_log", type=str)
@click.option('--rnn_step', default=10)  # only for benchmarks with long horizon
def run(env,
        seed,
        weights,
        gamma,
        replay_buffer_size,
        hidden_size,
        value_lr,
        policy_lr,
        train_episodes,
        batch_size,
        update_freq,
        eval_freq,
        use_exp,
        soft_tau,
        log_name,
        tensorboard_dir,
        rnn_step
        ):
    if 'Platform' in env:
        max_steps = 250
    elif 'Goal' in env:
        max_steps = 150

    save_path = '/data/data0/xuehui/workspace/csh/cp'
    state_dim = env.observation_space.shape[0]
    action_discrete_dim, action_continuous_dim = env.action_space.spaces[0].n, env.action_space.spaces[1].shape[0]
    device = "cuda"

    soft_q_net1 = PASAC_QNetwork_MLP(max_steps, state_dim, action_discrete_dim, action_continuous_dim,
                                     hidden_size=hidden_size, batch_size=batch_size).to(device)
    soft_q_net2 = PASAC_QNetwork_MLP(max_steps, state_dim, action_discrete_dim, action_continuous_dim,
                                     hidden_size=hidden_size, batch_size=batch_size).to(device)
    target_soft_q_net1 = PASAC_QNetwork_MLP(max_steps, state_dim, action_discrete_dim, action_continuous_dim,
                                            hidden_size=hidden_size, batch_size=batch_size).to(device)
    target_soft_q_net2 = PASAC_QNetwork_MLP(max_steps, state_dim, action_discrete_dim, action_continuous_dim,
                                            hidden_size=hidden_size, batch_size=batch_size).to(device)
    policy_net = PASAC_PolicyNetwork_MLP(state_dim,
                                         max_steps,
                                         action_discrete_dim,
                                         action_continuous_dim,
                                         hidden_size=hidden_size,
                                         batch_size=batch_size).to(device)
    models = {'policy': policy_net,
              'value1': soft_q_net1, 'target_value1': target_soft_q_net1,
              'value2': soft_q_net2, 'target_value2': target_soft_q_net2}


    for name, model in models.items():
            complete_load_name = save_path + "_" + name + ".pt"
            model.load_state_dict(state_dict=torch.load(complete_load_name))
            print("[loaded " + complete_load_name + ']')

