opyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import os
import sys
import argparse
import pprint

import numpy as np
import torch

from create_envs import create_train_env, create_eval_env
import vdn_r2d2
import iql_r2d2
import common_utils
import rela
from eval import evaluate
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--method", type=str, default="vdn")

    # game settings
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--greedy_extra", type=int, default=0)
    parser.add_argument("--num_player", type=int, default=2)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-4, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--rnn_hid_dim", type=int, default=512)

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument(
        "--batchsize", type=int, default=12,
    )
    parser.add_argument("--num_epoch", type=int, default=5000)
    #parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--epoch_len", type=int, default=1000)
   # parser.add_argument("--epoch_len", type=int, default=500)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settings
    #parser.add_argument("--burn_in_frames", type=int, default=100)
    parser.add_argument("--burn_in_frames", type=int, default=80000)
    parser.add_argument("--replay_buffer_size", type=int, default=2 ** 20)
    #parser.add_argument("--replay_buffer_size", type=int, default=2 ** 8)
    parser.add_argument(
        "--priority_exponent", type=float, default=0.6, help="prioritized replay alpha",
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.4, help="prioritized replay beta",
    )
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=40, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=20)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.4)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    #parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--act_device", type=str, default="cpu")
    parser.add_argument("--actor_sync_freq", type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    saver = common_utils.TopkSaver(args.save_dir, 10)
    common_utils.set_all_seeds(args.seed)
    game_info = utils.get_game_info(args.num_player, args.greedy_extra)

    agent = vdn_r2d2.R2D2Agent(
        args.multi_step,
        args.gamma,
        0.9,
        args.train_device,
        game_info["input_dim"],
        args.rnn_hid_dim,
        game_info["num_action"],
    )

    agent2 = vdn_r2d2.R2D2Agent(
            args.multi_step,
            args.gamma,
            0.9,
            args.train_device,
            game_info["input_dim"],
            args.rnn_hid_dim,
            game_info["num_action"],
        )

    """LOAD MODELS"""
    path1 = "exps/op1/model3.pthw"
    path2 = "exps/op2/model1.pthw"

    saver.load_weights(agent, path1)
    saver.load_weights(agent2, path2)

    # eval is always in IQL fashion
    eval_agents = []
    eval_lockers = []
    for _ in range(args.num_player):
        ea = iql_r2d2.R2D2Agent(
            1,
            0.99,
            0.9,
            "cpu",
            game_info["input_dim"],
            args.rnn_hid_dim,
            game_info["num_action"],
        )
        locker = rela.ModelLocker([ea], "cpu")
        eval_agents.append(ea)
        eval_lockers.append(locker)

    agent = agent.to(args.train_device)
    agent2 = agent2.to(args.train_device)

    # for i in range(args.num_player):
    #     eval_lockers[i].update_model(agent)
    eval_lockers[0].update_model(agent)
    eval_lockers[1].update_model(agent2)
    # eval_seed = (args.seed * 1000) % 7777777
    eval_seed = 1
    score, perfect, _, _ = evaluate(
        eval_lockers,
        1000,
        eval_seed,
        0,
        args.num_player,
        args.eval_bomb,
        args.greedy_extra,
    )
    print("Eval score: %.4f, perfect: %.2f" % (score, perfect * 100))

