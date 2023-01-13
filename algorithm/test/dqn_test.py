import argparse
import os
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
import env_design.env1 as env1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=1.0)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-collect', type=int, default=100)
    parser.add_argument('--update-per-step', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256,128,128])
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='algorithm/log')
    parser.add_argument('--render', type=float, default=1/700)
    parser.add_argument(
        '--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--save_buffer_name', type=str, default=None)
    return parser.parse_args()


def test_dqn(arg,norm_data,ori_rank,args=get_args()):
    env = env1.Env_One_Point(arg,norm_data,ori_rank)  # 单独环境的实例化
    args.state_shape = env.observation_space.shape[0] or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # train_envs = gym.make(args.task)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv(
        [lambda: env1.Env_One_Point(arg,norm_data,ori_rank) for _ in range(args.training_num)]
    )
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: env1.Env_One_Point(arg,norm_data,ori_rank) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    # Q_param = {"hidden_sizes": args.dueling_q_hidden_sizes}
    # V_param = {"hidden_sizes": args.dueling_v_hidden_sizes}
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        # dueling_param=(Q_param, V_param)
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
        is_double=False
    )
    if os.path.exists(os.path.join(args.logdir,'dqn/policy.pth')):
        policy.load_state_dict(torch.load(os.path.join(args.logdir,'dqn/policy.pth')))
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    # train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir,'dqn')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def train_fn(epoch, env_step):
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                40000 * (0.5 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.5 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(
                n_episode=args.test_num, render=args.render
            )
        rew = result["rews"].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")

    watch()


if __name__ == '__main__':
    test_dqn(get_args())