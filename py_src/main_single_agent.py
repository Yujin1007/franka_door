import sys
sys.path.append('/home/kist-robot2/Franka/franka_door')
import numpy as np
import torch
import gym
import argparse
import json
import os
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqc import structures, DEVICE
from tqc.trainer import Trainer
from tqc.structures import Actor, Critic, RescaleAction
from tqc.functions import eval_policy
import fr3Env
import tools
import torch.nn as nn
import torch.nn.functional as F

HEURISTIC = 0
RL = 1


def main(PATH, TRAIN, RENDERING):
    env = fr3Env.door_env2()
    env.env_rand =False
    env.rendering = RENDERING
    PLANNING_MODE = RL
    # TRAIN = True
    # PLANNING_MODE = HEURISTIC
    env.train = TRAIN
    # max_timesteps = 1e3
    max_timesteps = 1e6

    max_episode = 1e4
    batch_size = 16
    policy_kwargs = dict(n_critics=5, n_quantiles=25)
    save_freq = 1e5
    models_dir = PATH
    pretrained_model_dir = models_dir #+ "8.0/" # 6.0 : 6.4 , 5: 5.7
    # pretrained_model_dir = models_dir + "10.0/" # 6.0 : 6.4 , 5: 5.7
    episode_data = []
    timestep_data = []
    save_flag = False

    state_dim = env.observation_space.shape
    action_dim1 = env.rotation_action_space.shape[0]
    action_dim2 = env.force_action_space.shape[0]

    replay_buffer1 = structures.ReplayBuffer((env.len_hist,state_dim), action_dim1+action_dim2)
    actor1 = Actor(state_dim, action_dim1+action_dim2, IsDemo).to(DEVICE)

    critic1 = Critic(state_dim, action_dim1+action_dim2, policy_kwargs["n_quantiles"], policy_kwargs["n_critics"],IsDemo).to(DEVICE)
    critic_target1 = copy.deepcopy(critic1)

    trainer1 = Trainer(actor=actor1,
                      critic=critic1,
                      critic_target=critic_target1,
                      top_quantiles_to_drop=2,
                      discount=0.99,
                      tau=0.005,
                      target_entropy=-np.prod(env.rotation_action_space.shape).item())


    episode_return_rotation = 0
    episode_return_force = 0
    episode_timesteps = 0
    episode_num = 0

    if TRAIN:
        state = env.reset(PLANNING_MODE)
        # trainer.load(pretrained_model_dir)

        actor1.train()
        actor2.train()
        return_rotation_max = 100
        return_force_max = 100
        for t in range(int(max_timesteps)):

            action_rotation = actor1.select_action(state)
            action_force = actor2.select_action(state)


            next_state, reward_rotation, reward_force, done, _ = env.step(action_rotation, action_force)

            episode_timesteps += 1

            replay_buffer1.add(state, action_rotation, next_state, reward_rotation, done)
            replay_buffer2.add(state, action_force, next_state, reward_force, done)

            state = next_state
            episode_return_rotation += reward_rotation
            episode_return_force += reward_force

            # Train agent after collecting sufficient data
            if t >= batch_size:
                if IsDemo:
                    trainer1.train_with_demo(replay_buffer1, replay_buffer1_expert, batch_size)
                    trainer2.train_with_demo(replay_buffer2, replay_buffer2_expert, batch_size)

                else:
                    trainer1.train(replay_buffer1, batch_size)
                    trainer2.train(replay_buffer2, batch_size)
            if (t + 1) % save_freq == 0:
                save_flag = True
            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
                    f"Reward R: {episode_return_rotation:.3f} Reward F: {episode_return_force:.3f}" f"Best R: {return_rotation_max:.3f} Best F: {return_force_max:.3f}")


                if t > save_freq and episode_return_rotation > return_rotation_max and episode_return_force > return_force_max:
                    return_rotation_max = episode_return_rotation
                    return_force_max = episode_return_force
                    path1 = models_dir + "best/rotation/"
                    path2 = models_dir + "best/force/"
                    trainer1.save(path1)
                    trainer2.save(path2)
                # Reset environment
                state = env.reset(PLANNING_MODE)
                episode_data.append([episode_num, episode_timesteps, episode_return_rotation, episode_return_force])
                episode_return_rotation_accum += episode_return_rotation
                episode_return_force_accum += episode_return_force
                episode_cnt += 1

                episode_return_rotation = 0
                episode_return_force = 0
                episode_timesteps = 0
                episode_num += 1
            if (t + 1) % 10000 == 0:
                timestep_data.append(
                    [episode_return_rotation_accum / episode_cnt, episode_return_force_accum / episode_cnt])
                episode_return_rotation_accum = 0
                episode_return_force_accum = 0
                episode_cnt = 0
                np.save(models_dir + "avg_reward.npy", timestep_data)

            if save_flag:
                path1 = models_dir + str((t + 1) // save_freq) + "/rotation/"
                path2 = models_dir + str((t + 1) // save_freq) + "/force/"
                trainer1.save(path1)
                trainer2.save(path2)

                np.save(models_dir + "reward.npy", episode_data)
                save_flag = False

    else:
        pretrained_model_dir1 = pretrained_model_dir+"rotation/"
        pretrained_model_dir2 = pretrained_model_dir+"force/"
        trainer1.load(pretrained_model_dir1)
        actor1.eval()
        critic1.eval()
        actor1.training = False

        trainer2.load(pretrained_model_dir2)
        actor2.eval()
        critic2.eval()
        actor2.training = False

        # reset_agent.training = False
        num_ep = 2
        force_data = []
        reward_data = []
        # env.episode_number = 3
        # df = pd.read_csv("/home/kist-robot2/Downloads/obs_real.csv")
        # states = df.to_numpy(dtype=np.float32)
        for _ in range(num_ep):
            state = env.reset(PLANNING_MODE)
            done = False
            step_cnt = 0
            episode_return_rotation = 0
            episode_return_force = 0

            while not done:

                step_cnt += 1
                action_rotation = actor1.select_action(state)
                action_force = actor2.select_action(state)

                next_state, reward_rotation, reward_force, done, _ = env.step(action_rotation, action_force)
                force_data.append(env.force)
                reward_data.append([reward_rotation, reward_force])

                state = next_state
                episode_return_rotation += reward_rotation
                episode_return_force += reward_force

            # np.save("./data/torque_hybrid.npy", env.torque_data)
            # print(
            #     f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
            #     f"Reward R: {episode_return_rotation:.3f} Reward F: {episode_return_force:.3f}")
            print(
                f"Reward R: {episode_return_rotation:.3f} Reward F: {episode_return_force:.3f}")
            print("time:",env.time_done, "  contact:",env.contact_done, "  bound:",env.bound_done,
                  "  goal:", env.goal_done)
            print(env.door_angle)
            # np.save("./data/heuristic_rpy_data.npy",env.rpyfromvalve_data)
            # np.save("./data/heuristic_xyz_data.npy",env.xyzfromvalve_data )

            fig, axs = plt.subplots(3, 2, figsize=(8, 6))
            axs[0, 0].plot([sublist[0] for sublist in env.command_data])
            axs[0, 0].set_title("droll", pad=20)

            axs[0, 1].plot([sublist[1] for sublist in env.command_data])
            axs[0, 1].set_title("dpitch", pad=20)

            axs[1, 0].plot([sublist[2] for sublist in env.command_data])
            axs[1, 0].set_title("roll", pad=20)

            axs[1, 1].plot([sublist[3] for sublist in env.command_data])
            axs[1, 1].set_title("pitch", pad=20)

            axs[2, 0].plot([sublist[4] for sublist in env.command_data])
            axs[2, 0].set_title("force gain", pad=20)

            axs[2, 1].plot([sublist[5] for sublist in env.command_data])
            axs[2, 1].set_title("R force gain", pad=20)

            fig2, axs2 = plt.subplots(1, 2, figsize=(8, 6))
            axs2[0].plot([sublist[0] for sublist in reward_data])
            axs2[0].set_title("reward_rotation", pad=20)

            axs2[1].plot([sublist[1] for sublist in reward_data])
            axs2[1].set_title("reward_force", pad=20)
            # plt.plot(force_data,  linestyle='-', color='b')
            # # plt.title(env.friction)
            plt.show()
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
