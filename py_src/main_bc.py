import numpy as np
import argparse
import json
import os
import copy
import matplotlib.pyplot as plt
import fr3Env
import torch
import torch.nn as nn
import torch.optim as optim
from tqc import structures
from bc.structure import BC_Actor, ExpertDataset
from torch.utils.data import DataLoader, Dataset
from Classifier import Classifier
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
HEURISTIC = 0
RL = 1
def save_replay_buffer(buffer, filename):
    with open(filename, 'wb') as f:
        pickle.dump(buffer, f)

def load_replay_buffer(filename):
    with open(filename, 'rb') as f:
        buffer = pickle.load(f)
    return buffer
def main(PATH, COLLECT, TRAIN, RENDER):
    env = fr3Env.door_env2()
    Collect_Data = COLLECT
    env.rendering = RENDER
    state_dim = env.observation_space.shape
    action_dim1 = env.rotation_action_space.shape[0]
    action_dim2 = env.force_action_space.shape[0]

    # input_size = dataset.observations[0].shape[1]
    # output_size = dataset.actions[0].shape[0]
    model = BC_Actor(state_dim, action_dim1 + action_dim2).to(DEVICE)


    if TRAIN:
        if Collect_Data:
            num_epochs = int(1e4)
            replay_buffer1 = structures.ReplayBuffer((env.len_hist, state_dim), action_dim1, max_size=int(1e4))
            replay_buffer2 = structures.ReplayBuffer((env.len_hist, state_dim), action_dim2, max_size=int(1e4))
            state = env.reset(HEURISTIC)
            step_cnt = 0
            for epoch in range(num_epochs):
                action_rotation = np.zeros(2)
                action_force = env.controller.get_force()
                action_force = np.array([action_force])
                next_state, reward_rotation, reward_force, done, _ = env.step(action_rotation, action_force)
                replay_buffer1.add(state, action_rotation, next_state, reward_rotation, done)
                replay_buffer2.add(state, action_force, next_state, reward_force, done)
                step_cnt += 1
                state = next_state
                if done:
                    state = env.reset(HEURISTIC)
                    print(step_cnt)
            expert_path = "./log/expert/"
            os.makedirs(expert_path, exist_ok=True)
            if not os.path.exists(expert_path):
                os.makedirs(expert_path)
            save_replay_buffer(replay_buffer1, expert_path + 'replay_buffer_rotation.pkl')
            save_replay_buffer(replay_buffer2, expert_path + 'replay_buffer_force.pkl')


        expert1_path = './log/expert/replay_buffer_rotation.pkl'
        expert2_path = './log/expert/replay_buffer_force.pkl'

        dataset = ExpertDataset(expert1_path, expert2_path)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        num_epochs = 100
        for epoch in range(num_epochs):
            for observations, actions in dataloader:
                optimizer.zero_grad()
                outputs = model(observations.to(DEVICE))
                loss = criterion(outputs, actions.to(DEVICE))
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        torch.save(model.state_dict(), PATH+'model.pth')

    else:
        PATH = str(PATH)
        # self.critic.load_state_dict(torch.load(filename + "_critic"))
        state_dict=torch.load(PATH + "model.pth")
        model.load_state_dict(state_dict)
        model.eval()
        num_ep = 16
        for _ in range(num_ep):
            state = env.reset()
            done = False
            step_cnt = 0
            episode_return_rotation = 0
            episode_return_force = 0
            force_data = []
            while not done:

                step_cnt += 1
                action = model.select_action(state)
                action_rotation = action[:2]
                action_force = action[2]

                next_state, reward_rotation, reward_force, done, _ = env.step(action_rotation, action_force)
                force_data.append(env.force)

                state = next_state
                episode_return_rotation += reward_rotation
                episode_return_force += reward_force

            print(
                f"Reward R: {episode_return_rotation:.3f} Reward F: {episode_return_force:.3f}")
            print("time:",env.time_done, "  contact:",env.contact_done, "  bound:",env.bound_done,
                  "  goal:", env.goal_done)
            plt.plot(force_data,  linestyle='-', color='b')
            plt.title(env.friction)
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(description="valve_env2!")
    parser.add_argument("--path", help="data load path", default=" ./log/bc/0708_1/")
    parser.add_argument("--collect", help="1->collect buffer,  0->use buffer", type=int, default=1)
    parser.add_argument("--train", help="0->test,  1->train", type=int, default=1)
    parser.add_argument("--render", help="0->no rendering,  1->rendering", type=int, default=0)
    # parser.add_argument("--offline", help="0->no offline data,  1->with offline data", type=int, default=0)
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['description'] = parser.description
    os.makedirs(args.path, exist_ok=True)
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    # Print Arguments
    print("------------ Arguments -------------")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    print("------------------------------------")

    main(PATH=args.path, COLLECT=args.collect, TRAIN=args.train, RENDER=args.render)
