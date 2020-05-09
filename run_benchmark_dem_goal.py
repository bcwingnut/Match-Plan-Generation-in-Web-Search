import click
import os
import warnings
from Benchmarks.experiment_benchmark import train_dem_mlp
import torch


@click.command()
@click.option('--env', default='Goal-v0')  # = Platform-v0 / Goal-v0
@click.option('--seed', default=7)
@click.option('--log_name', default="Goal_psac3")
@click.option('--weights', default=[1., -1, 0.])  # [reward design]
@click.option('--gamma', default=0.99)
@click.option('--replay_buffer_size', default=50000)
# [network]
@click.option('--hidden_size', default=2048)
@click.option('--value_lr', default=0.0009270236969863404)
@click.option('--policy_lr', default=0.0003207605221990806)
# [training]
@click.option('--train_episodes', default=500000)
@click.option('--batch_size', default=128)
@click.option('--update_freq', default=1)
@click.option('--eval_freq', default=10)
@click.option('--use_exp', default=True)  # True = exp, False = Sample
@click.option('--soft_tau', default=0.005454151630279901)
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
    print('\n'.join(['%s:%s' % item for item in locals().items()]))

    debug = {
            'log': log_name,
            'tensorboard_dir': tensorboard_dir,
            #'name_query': "PROD_2019-09-06_WebPrecision_Draco_25682_saaswarmuplego.tsv",
            'name_query':"PROD_2019-03-20_WebPrecision_Draco_25178_saaswarmuplego.tsv",
            #'name_embedding': "PROD_2019-09-06_WebPrecision_Draco_25682_saaswarmuplego.encoded.tsv",
            'name_embedding': "output.encoded.tsv",
            'debug': True,
            'download': True,
            'env': env,

            'print_query': False,
            'print_step': False,
            'plot': False,  # verbose, print sas'
            'plot_base': False,
            'print_freq': 32,  # print log per 'print_freq' episodes
            'eval_csv_output': False,  # generate a csv that contains detailed information if True

            'eval': False,
            'imitate': False,
            'use_embedding': True,
            'use_act_num': False,  # use history action as state information
            'fix_query': False,
            'sampling': True,

            'use_demonstration': False,
            'load_demonstration': True,
            'save_demonstration': False,
            'demonstration_path': '/data/data0/xuehui/workspace/csh/de/Goal_1.pkl',
            'demonstration_buffer_size': int(1e5),
            'demonstration_ratio_step': 2e-5,
            'demonstration_number': int(1e5),
            "behavior_cloning": True,
            "bcloss_weight": 0.1,
            "discrete_bcloss_weight": 5,
            'pretrain': False,
            'pretrain_episodes': 50000,
            'ep_punishment': 0.01,

            'save_model': True,  # save checkpoints or not
            'load_model': False,  # load checkpoints at the beginning or not, 'load_filename' should be assigned
            'checkpoint_save_path': '/data/data0/xuehui/workspace/csh/cp',
            'replay_buffer_save_path': '/data/data2/csh/buffer/psac2_ep20000_20200402124642.pk',
            'load_filename': 'psac2-20000-Apr-02-2020 12:46:39',  # prefix of the checkpoint
            'save_freq': 10000,  # save a model per 'save_freq' episodes
            'forward_time': True,

            'L2_norm': 0,
            'replay_buffer': 'lp3',
            'demonstration_buffer': 'lp2',
            'capacity_distribution': 'uniform',
            'use_log': False,
            'ep_punishment': 0.01
            # n for normal, p for prioritized(lifei), s for stratefied, l for sac_lstm, lp for prioritized_sac_lstm
            }

    print('=================debug parameters=================')
    print('\n'.join(['%s:%s' % item for item in debug.items()]))
    print('=================debug parameters=================')

    if 'Platform' in env:
        max_steps = 250
    elif 'Goal' in env:
        max_steps = 150
    else:
        raise NotImplementedError
    
    seed_torch(seed)
    warnings.filterwarnings("ignore")
    # train for PASAC_LSTM, train_mlp for PASAC_MLP
    train_dem_mlp(log_name, env, debug,
          seed, max_steps, train_episodes,
          batch_size, update_freq, eval_freq,
          weights, gamma, replay_buffer_size,
          hidden_size, value_lr, policy_lr,
          soft_tau=soft_tau,
          use_exp=True,
          use_nni=False)


def seed_torch(seed=1029):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('******PID:' + str(os.getpid()) + '******')

    gpu_id = 1  # change here to alter GPU id
    print('GPU id:'+str(gpu_id))
    with torch.cuda.device(gpu_id):
        run()

    print('============Done============')
    # nohup python -u run_benchmark.py >log/benchmark.log 2>&1 &
