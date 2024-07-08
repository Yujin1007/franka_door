import numpy as np
import torch
import gym
import argparse
import os
import copy
import argparse
import json

import matplotlib.pyplot as plt
from pathlib import Path
# from stable_baselines3.common.policies import BasePolicy, register_policy
import pandas as pd
from tqc import structures, DEVICE
from tqc.trainer import Trainer
from tqc.structures import Actor, Critic, RescaleAction
from tqc.functions import eval_policy
import fr3Env
import tools
import torch.nn as nn
import torch.nn.functional as F
from Classifier import Classifier
import csv
HEURISTIC = 0
RL = 1
def save_csv_data(file_name, file_path, data):
    os.makedirs(file_path, exist_ok=True)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    mode = 'a' if os.path.exists(file_path + file_name) else 'w'
    with open(file_path + file_name, mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    data = []
    return data

def main(PATH, TRAIN, OFFLINE, RENDERING):

    # env = fr3Env.valve_env(TRAIN=TRAIN, RENDER=RENDERING, OBJ=OBJ)

    env = fr3Env.door_env2()
    env.train = TRAIN
    env.env_rand =True
    env.rendering = RENDERING
    if OFFLINE == 0:
        IsDemo = False
    else:
        IsDemo = True
    max_timesteps = 1e6
    max_episode = 1e4
    batch_size = 256
    policy_kwargs = dict(n_critics=5, n_quantiles=25)
    save_freq = 1e5
    # models_dir = PATH
    # save_file_path = "./data/done_data_0529_10.csv"

    # pretrained_model_dir = "./log/rollpitch_acc2/25.0/"
    pretrained_model_dir = PATH
    save_file_path = pretrained_model_dir + "data/"
    episode_data = []
    save_flag = False
    state_dim = env.observation_space.shape
    action_dim1 = env.rotation_action_space.shape[0]
    action_dim2 = env.force_action_space.shape[0]

    # replay_buffer1 = structures.ReplayBuffer((env.len_hist, state_dim), action_dim1, max_size=int(1e5))
    actor1 = Actor(state_dim, action_dim1, IsDemo).to(DEVICE)

    critic1 = Critic(state_dim, action_dim1, policy_kwargs["n_quantiles"], policy_kwargs["n_critics"], IsDemo).to(
        DEVICE)
    critic_target1 = copy.deepcopy(critic1)

    trainer1 = Trainer(actor=actor1,
                       critic=critic1,
                       critic_target=critic_target1,
                       top_quantiles_to_drop=2,
                       discount=0.99,
                       tau=0.005,
                       target_entropy=-np.prod(env.rotation_action_space.shape).item())

    # replay_buffer2 = structures.ReplayBuffer((env.len_hist, state_dim), action_dim2, max_size=int(1e5))
    actor2 = Actor(state_dim, action_dim2, IsDemo).to(DEVICE)

    critic2 = Critic(state_dim, action_dim2, policy_kwargs["n_quantiles"], policy_kwargs["n_critics"], IsDemo).to(
        DEVICE)
    critic_target2 = copy.deepcopy(critic2)

    trainer2 = Trainer(actor=actor2,
                       critic=critic2,
                       critic_target=critic_target2,
                       top_quantiles_to_drop=2,
                       discount=0.99,
                       tau=0.005,
                       target_entropy=-np.prod(env.force_action_space.shape).item())

    episode_return = 0
    episode_timesteps = 0
    episode_num = 0

    pretrained_model_dir1 = pretrained_model_dir + "rotation/"
    pretrained_model_dir2 = pretrained_model_dir + "force/"
    trainer1.load(pretrained_model_dir1)
    actor1.eval()
    critic1.eval()
    actor1.training = False

    trainer2.load(pretrained_model_dir2)
    actor2.eval()
    critic2.eval()
    actor2.training = False


    data_fail = []
    data_success = []
    data_qpos_s = []
    data_xpos_s = []
    data_env_s = []
    data_qpos_f = []
    data_xpos_f = []
    data_env_f = []
    # reset_agent.training = False
    num_ep = 2
    success_cnt = 0
    fail_cnt = 0
    # env.episode_number = 3
    # df = pd.read_csv("/home/kist-robot2/Downloads/obs_real.csv")
    # states = df.to_numpy(dtype=np.float32)
    for ep in range(num_ep):
        if ep == 0:
            PLANNING_MODE = RL
        else:
            PLANNING_MODE = HEURISTIC
        state = env.reset(PLANNING_MODE)
        done = False
        episode_return = 0
        step_cnt = 0
        episode_return_rotation = 0
        episode_return_force = 0
        while not done:
            action_rotation = actor1.select_action(state)
            action_force = actor2.select_action(state)

            next_state, reward_rotation, reward_force, done, _ = env.step(action_rotation, action_force)

            state = next_state
            episode_return_rotation += reward_rotation
            episode_return_force += reward_force
        if ep == 0:
            RL_manipulability = env.manipulability_data.copy()
        else:
            HEURISTIC_manipulability = env.manipulability_data.copy()
        # np.save("./data/torque_hybrid.npy", env.torque_data)
        # RL_handle_angle = env.handle_angle
        # RL_done = env.done_check
        # RL_q_end = env.data.qpos[:env.dof].copy()
        # RL_x_end = env.x_end.copy()
        # RL_manipulability = env.manipulability_data.copy()
        # env.heuristic_compare()
        # HEURISTIC_handle_angle = env.handle_angle
        # HEURISTIC_done = env.done_check
        # HEURISTIC_q_end = env.data.qpos[:env.dof].copy()
        # HEURISTIC_x_end = env.x_end.copy()
        # HEURISTIC_manipulability = env.manipulability_data.copy()

    plt.plot([sublist[0] for sublist in RL_manipulability],[sublist[1] for sublist in RL_manipulability], linestyle='-', color='r', label='RL Manipulability')
    plt.plot([sublist[0] for sublist in HEURISTIC_manipulability],[sublist[1] for sublist in HEURISTIC_manipulability], linestyle='-', color='b', label='Heuristic Manipulability')
    # plt.plot(RL_manipulability, linestyle='-', color='r', label='RL Manipulability')
    # plt.plot(HEURISTIC_manipulability, linestyle='-', color='b', label='Heuristic Manipulability')

    plt.title(f"Manipulability")
    plt.xlabel('X-axis Label')  # Replace with appropriate label
    plt.ylabel('Y-axis Label')  # Replace with appropriate label
    plt.legend()  # Add legend to the plot
    plt.show()
        #
        # plt.grid(True)  # Add grid for better readability
        # plt.show()
        # if abs(RL_handle_angle)>= abs(HEURISTIC_handle_angle):
        #     data_success.append([RL_handle_angle,RL_done, sum(RL_manipulability),HEURISTIC_handle_angle, HEURISTIC_done,sum(HEURISTIC_manipulability), env.model.body_pos[env.bid],env.model.body_quat[env.bid], env.scale, env.friction])
        #     data_qpos_s.append([env.q_reset[:env.dof],RL_q_end, HEURISTIC_q_end])
        #     data_xpos_s.append([env.x_reset, RL_x_end, HEURISTIC_x_end, RL_handle_angle, HEURISTIC_handle_angle])
        #     data_env_s.append([env.model.body_pos[env.bid],env.model.body_quat[env.bid], env.scale, env.friction])
        #     success_cnt += 1
        # else:
        #     data_fail.append(
        #         [RL_handle_angle, RL_done, sum(RL_manipulability), HEURISTIC_handle_angle, HEURISTIC_done,
        #          sum(HEURISTIC_manipulability), env.model.body_pos[env.bid], env.model.body_quat[env.bid], env.scale,
        #          env.friction])
        #
        #     data_qpos_f.append([env.q_reset[:env.dof], RL_q_end, HEURISTIC_q_end])
        #     data_xpos_f.append(
        #         [env.x_reset, RL_x_end, HEURISTIC_x_end, RL_handle_angle, HEURISTIC_handle_angle])
        #     data_env_f.append([env.model.body_pos[env.bid], env.model.body_quat[env.bid], env.scale, env.friction])
        #     fail_cnt += 1
        # # print("episode :", env.episode_number, "handle angle :", np.round(env.handle_angle,3), " episode return:", episode_return)
        # #
        # # print("time:",env.time_done, "  contact:",env.contact_done, "  bound:",env.bound_done,
        # #       "  goal:", env.goal_done)
        # # print([RL_handle_angle, RL_done, HEURISTIC_handle_angle, HEURISTIC_done])
        # if ep % 20 == 19:
        #     data_fail = save_csv_data("compare_data.csv", save_file_path+"fail/", data_fail)
        #     data_qpos_f = save_csv_data("qpos.csv", save_file_path + "fail/", data_qpos_f)
        #     data_xpos_f = save_csv_data("xpos.csv", save_file_path + "fail/", data_xpos_f)
        #     data_env_f = save_csv_data("env.csv", save_file_path + "fail/", data_env_f)
        #
        #     data_success = save_csv_data("compare_data.csv", save_file_path + "success/", data_success)
        #     data_qpos_s = save_csv_data("qpos.csv", save_file_path + "success/", data_qpos_s)
        #     data_xpos_s = save_csv_data("xpos.csv", save_file_path + "success/", data_xpos_s)
        #     data_env_s = save_csv_data("env.csv", save_file_path + "success/", data_env_s)
        #
        #     print(f"Epsode : {ep} Success cnt : {success_cnt}  Fail cnt : {fail_cnt}")
        # with open(save_file_path, 'w', newline='') as file:
        #     # Step 4: Create a csv.writer object
        #     writer = csv.writer(file)
        #
        #     writer.writerows(data)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", help="description")
    parser.add_argument("--path", help="data load path", default=" ./log/0607_1/")
    parser.add_argument("--train", help="0->test,  1->train", type=int, default=1)
    parser.add_argument("--render", help="0->no rendering,  1->rendering", type=int, default=0)
    parser.add_argument("--offline", help="0->no offline data,  1->with offline data", type=int, default=0)
    args = parser.parse_args()
    args_dict = vars(args)
    if args.train == 1:
        os.makedirs(args.path, exist_ok=True)
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        with open(args.path+'args_and_notes.json', 'w') as f:
            json.dump(args_dict, f, indent=4)
    # Print Arguments
    print("------------ Arguments -------------")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    print("------------------------------------")

    main(PATH=args.path, TRAIN=args.train, OFFLINE=args.offline, RENDERING=args.render)
