import numpy as np
import torch
import gym
import argparse
import os
import copy
from pathlib import Path
from stable_baselines3.common.policies import BasePolicy, register_policy
import csv
from tqc import structures, DEVICE
from tqc.trainer import Trainer
from tqc.structures import Actor, Critic, RescaleAction
from tqc.functions import eval_policy
from fr3Env import fr3_smooth_start
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from Classifier import Classifier
# class Classifier(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 256)
#         self.fc_ = nn.Linear(256, 256)
#         self.fc4 = nn.Linear(256, output_size)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc_(x))
#         x = F.relu(self.fc4(x))
#         return x

def main():
    # --- Init ---

    # env = fr3_valve_size()
    env = fr3_smooth_start()
    policy_kwargs = dict(n_critics=5, n_quantiles=25)
    # models_dir = "./log/new_loss_1/"
    models_dir = "./log/smooth_start3/"

    models_subdir=models_dir+"8.0/"

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim, action_dim, policy_kwargs["n_quantiles"], policy_kwargs["n_critics"]).to(DEVICE)
    critic_target = copy.deepcopy(critic)

    ## saving
    trainer = Trainer(actor=actor,
                      critic=critic,
                      critic_target=critic_target,
                      top_quantiles_to_drop=2,
                      discount=0.99,
                      tau=0.005,
                      target_entropy=-np.prod(env.action_space.shape).item())
    trainer.load(models_subdir)


    tmp = copy.deepcopy(actor.net.fc0.weight)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/actor/w1.csv", header=False, index=False)
    tmp = copy.deepcopy(actor.net.fc0.bias)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/actor/b1.csv", header=False, index=False)

    tmp = copy.deepcopy(actor.net.fc1.weight)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/actor/w2.csv", header=False, index=False)
    tmp = copy.deepcopy(actor.net.fc1.bias)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/actor/b2.csv", header=False, index=False)

    tmp = copy.deepcopy(actor.net.last_fc.weight)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/actor/w3.csv", header=False, index=False)
    tmp = copy.deepcopy(actor.net.last_fc.bias)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/actor/b3.csv", header=False, index=False)




    tmp = copy.deepcopy(env.classifier_cclk.fc1.weight)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_cclk/w1.csv", header=False, index=False)
    tmp = copy.deepcopy(env.classifier_cclk.fc1.bias)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_cclk/b1.csv", header=False, index=False)

    tmp = copy.deepcopy(env.classifier_cclk.fc2.weight)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_cclk/w2.csv", header=False, index=False)
    tmp = copy.deepcopy(env.classifier_cclk.fc2.bias)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_cclk/b2.csv", header=False, index=False)

    tmp = copy.deepcopy(env.classifier_cclk.fc_.weight)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_cclk/w3.csv", header=False, index=False)
    tmp = copy.deepcopy(env.classifier_cclk.fc_.bias)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_cclk/b3.csv", header=False, index=False)

    tmp = copy.deepcopy(env.classifier_cclk.fc4.weight)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_cclk/w4.csv", header=False, index=False)
    tmp = copy.deepcopy(env.classifier_cclk.fc4.bias)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_cclk/b4.csv", header=False, index=False)


    tmp = copy.deepcopy(env.classifier_clk.fc1.weight)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_clk/w1.csv", header=False, index=False)
    tmp = copy.deepcopy(env.classifier_clk.fc1.bias)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_clk/b1.csv", header=False, index=False)

    tmp = copy.deepcopy(env.classifier_clk.fc2.weight)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_clk/w2.csv", header=False, index=False)
    tmp = copy.deepcopy(env.classifier_clk.fc2.bias)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_clk/b2.csv", header=False, index=False)

    tmp = copy.deepcopy(env.classifier_clk.fc_.weight)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_clk/w3.csv", header=False, index=False)
    tmp = copy.deepcopy(env.classifier_clk.fc_.bias)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_clk/b3.csv", header=False, index=False)

    tmp = copy.deepcopy(env.classifier_clk.fc4.weight)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_clk/w4.csv", header=False, index=False)
    tmp = copy.deepcopy(env.classifier_clk.fc4.bias)
    tmp = tmp.detach().cpu().numpy()
    pd.DataFrame(tmp).to_csv("./weight/classifier_clk/b4.csv", header=False, index=False)


    ## test saved weights
    # fc1_b = torch.load("./weight/b1")
    # fc1_w = torch.load("./weight/w1")
    # fc2_b = torch.load("./weight/b2")
    # fc2_w = torch.load("./weight/w2")
    # fc_mu_b = torch.load("./weight/b3")
    # fc_mu_w = torch.load("./weight/w3")
    #
    # actor.net.fc0.weight =fc1_w
    # actor.net.fc0.bias=fc1_b
    # actor.net.fc1.weight=fc2_w
    # actor.net.fc1.bias=fc2_b
    # actor.net.last_fc.weight=fc_mu_w
    # actor.net.last_fc.bias=fc_mu_b
    #
    # actor.net.fcs[0].weight=fc1_w
    # actor.net.fcs[0].bias=fc1_b
    # actor.net.fcs[1].weight=fc2_w
    # actor.net.fcs[1].bias=fc2_b

    numbers_array = []

    # Path to your CSV file
    csv_file = '/home/kist-robot2/Downloads/new_code/clik_obs_static.csv'

    # Open the CSV file and read line by line
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert each element in the row to integers and append to the list
            row = row[:91]
            numbers_array.append([float(num) for num in row])

    episode_return = 0

    actor.eval()
    critic.eval()
    actor.training = False
    # reset_agent.training = False
    num_ep = 5

    # print(actor.select_action(numbers_array[0]))
    # print("0.48236	-1.69255	3.66839	-0.984286	2.45705	-3.29748")
    for _ in range(num_ep):
        state = env.reset()
        done = False
        while not done:
            action = actor.select_action(state)

            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        print("episode :", env.episode_number, "goal angle :", env.required_angle, "handle angle :", env.handle_angle)

        print("time:",env.time_done, "  contact:",env.contact_done, "  bound:",env.bound_done,
              "  goal:", env.goal_done, "  reset:",env.reset_done)

if __name__ == "__main__":
    main()
