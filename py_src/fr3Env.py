#!/usr/bin/env python3
import sys
import time

import pandas as pd

sys.path.append('/home/kist-robot2/Franka/franka_valve')
import numpy as np
import sys
from numpy.linalg import inv
from build import controller

import mujoco
import gym
from gym import spaces
from random import random, randint, uniform
from scipy.spatial.transform import Rotation as R
from mujoco import viewer
from time import sleep
import tools
import rotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm


BODY = 1
JOINT = 3
GEOM = 5
MOTION_TIME_CONST = 10.
TASK_SPACE_TIME = 3+1+0.5

RL = 2
MANUAL = 1

RPY = False
XYZRPY = True

JOINT_CONTROL = 1
TASK_CONTROL = 2
HEURISTIC_CIRCULAR_CONTROL = 3
RL_CIRCULAR_CONTROL = 4
RL_CONTROL = 6



# class door_env1:
#     metadata = {"render_modes": ["human"], "render_fps": 30}
#
#     def __init__(self) -> None:
#         self.k = 7  # for jacobian calculation
#         self.dof = 9  # all joints (include gripper joint)
#         self.model_path = "../model/scene_door.xml"
#         self.model = mujoco.MjModel.from_xml_path(self.model_path)
#         self.data = mujoco.MjData(self.model)
#         mujoco.mj_step(self.model, self.data)
#         self.controller = controller.CController(self.k)
#         self.rendering = True
#         self.train = False
#         self.env_rand = False
#         self.len_hist = 10
#         self.downsampling = 100
#
#         self.observation_space = self._construct_observation_space()
#         self.rotation_action_space, self.force_action_space = self._construct_action_space()
#
#         self.viewer = None
#         self.q_range = self.model.jnt_range[:self.k]
#         self.qdot_range = np.array([[-2.1750, 2.1750], [-2.1750, 2.1750], [-2.1750, 2.1750], [-2.1750, 2.1750],
#                                     [-2.61, 2.61], [-2.61, 2.61], [-2.61, 2.61]])
#         self.tau_range = np.array([[0, 87], [0, 87], [0, 87], [0, 87], [0, 12], [0, 12], [0, 12]])
#         self.qdot_init = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#         self.q_init = [0, np.deg2rad(-60), 0, np.deg2rad(-90), 0, np.deg2rad(90), np.deg2rad(45), 0, 0, 0, 0]
#         # self.q_init = [0, np.deg2rad(-45), 0, np.deg2rad(-135), 0, np.deg2rad(90), np.deg2rad(45), 0, 0, 0, 0]
#         self.q_reset = [0, np.deg2rad(-60), 0, np.deg2rad(-90), 0, np.deg2rad(90), np.deg2rad(45), 0.03, 0.03, 0, 0]
#         self.episode_number = -1
#         desired_contact_list = ["finger_contact0", "finger_contact1",
#                                 "finger_contact2", "finger_contact3", "finger_contact4", "finger_contact5",
#                                 "finger_contact6", "finger_contact7",
#                                 "finger_contact8", "finger_contact9", "finger0_contact", "finger1_contact",
#                                 "door_handle",]
#         desired_contact_list_finger = ["finger_contact0", "finger_contact1",
#                                        "finger_contact2", "finger_contact3", "finger_contact4",
#                                        "finger_contact5", "finger_contact6", "finger_contact7",
#                                        "finger_contact8", "finger_contact9", ]
#         desired_contact_list_obj = ["door_handle"]
#         self.desired_contact_bid = tools.name2id(self.model, GEOM, desired_contact_list)
#         self.desired_contact_finger_bid = tools.name2id(self.model, GEOM, desired_contact_list_finger)
#         self.desired_contact_obj_bid = tools.name2id(self.model, GEOM, desired_contact_list_obj)
#
#         self.history_observation = np.zeros([self.len_hist * self.downsampling, self.observation_space.shape])
#         self.goal_latch_angle = -np.pi/2
#         self.latch_radius = 0.15
#         self.goal_door_angle = +np.pi/2
#         self.door_radius = 0.73 # 문고리 회전축~문 회전축까지
#         self.door_joint_idx = mujoco.mj_name2id(self.model, JOINT, "hinge")
#         self.latch_joint_idx = mujoco.mj_name2id(self.model, JOINT, "latch")
#         self.door_body_idx = mujoco.mj_name2id(self.model, BODY, "hinge_axis")
#         self.latch_body_idx = mujoco.mj_name2id(self.model, BODY, "latch_axis")
#         self.latch_info, self.door_info = self.env_information()
#
#
#     def reset(self, planning_mode):
#         self.control_mode = 0
#         # env_reset = True
#         self.episode_time_elapsed = 0.0
#         # self.torque_data = []
#         self.history_observation = np.zeros((self.len_hist * self.downsampling, self.observation_space.shape))
#         self.force = 0.0
#         duration = 0
#         # while env_reset:
#         self.episode_number += 1
#
#         self.start_time = self.data.time + 100
#         self.controller.initialize(planning_mode, self.latch_info, self.door_info, [self.goal_latch_angle, self.goal_door_angle])
#
#         self.data.qpos = self.q_init
#         self.data.qvel = self.qdot_init
#
#         self.episode_time = abs(
#             MOTION_TIME_CONST * abs(self.goal_door_angle) * self.door_radius) + abs(
#             MOTION_TIME_CONST * abs(self.goal_latch_angle) * self.latch_radius)
#
#         self.controller.read(self.data.time, self.data.qpos, self.data.qvel, self.model.opt.timestep)
#         self.controller.control_mujoco()
#
#         self.contact_done = False
#         self.bound_done = False
#         self.goal_done = False
#         self.action_rotation_pre = np.zeros(2)
#
#         obs = self._observation()
#         while self.control_mode != RL_CIRCULAR_CONTROL:
#             self.control_mode = self.controller.control_mode()
#
#
#             self.controller.read(self.data.time, self.data.qpos, self.data.qvel, self.model.opt.timestep)
#             self.controller.control_mujoco()
#             self._torque, _ = self.controller.write()
#             for i in range(self.dof - 1):
#                 self.data.ctrl[i] = self._torque[i]
#                 # self.data.ctrl[i] = 0.0
#             mujoco.mj_step(self.model, self.data)
#             # self.torque_data.append(self._torque[:7])
#
#             if duration == 10:
#                 duration = 0
#                 obs = self._observation()
#             done = self._done()
#             duration += 1
#             if done:
#                 break
#
#             if self.rendering:
#                 self.render()
#         self.door_angle = self.data.qpos[self.door_joint_idx]
#         self.door_angle_pre = self.door_angle
#         return obs
#
#     def step(self, action_rotation, action_force):
#
#         done = False
#         duration = 0
#         if action_force < -0.666:
#             action_force = -10
#         elif action_force < -0.333:
#             action_force = -1
#         elif action_force <= 0.333:
#             action_force = 0
#         else:
#             action_force = 1
#         self.force += action_force
#         self.force = np.clip(self.force, 0, abs(self.force))
#         self.door_angle = self.data.qpos[self.door_joint_idx]
#         while not done:
#             done = self._done()
#             self.control_mode = self.controller.control_mode()
#             self.controller.read(self.data.time, self.data.qpos, self.data.qvel, self.model.opt.timestep)
#
#             # --- RL controller input ---
#             if self.control_mode == RL_CIRCULAR_CONTROL:
#                 ddrollpitchdot_tmp = action_rotation * self.rotation_action_space.high
#                 duration += 1
#                 self.controller.put_action(ddrollpitchdot_tmp, action_force)
#             if duration == 10:
#                 break
#
#             self.controller.control_mujoco()
#             self._torque, _ = self.controller.write()
#             for i in range(self.dof - 1):
#                 self.data.ctrl[i] = self._torque[i]
#             mujoco.mj_step(self.model, self.data)
#             # self.torque_data.append(self._torque[:7])
#             # if abs(self._torque[3]) > 30:
#             #     print(self._torque[:7])
#             #     print(self._torque[:7])
#             # J = self.controller.get_jacobian()
#             # self.manipulability = tools.calc_manipulability(np.array(J))
#             if self.rendering:
#                 self.render()
#         obs = self._observation()
#         done = self._done()
#         reward_rotation, reward_force = self._reward(action_rotation, action_force)
#         info = self._info()
#         self.action_rotation_pre = action_rotation
#         self.door_angle_pre = self.door_angle
#
#         return obs, reward_rotation, reward_force, done, info
#
#     def _observation(self):
#         '''
#         observation 으로 필요한 것.
#         force + rotation 가 동일한 state 를 공유 하도록 하려면??
#         1. current q
#         2. current end-effector pose - kinematics 를 잘 해석하는 네트워크가 필요하겠다..?
#         -> 이것 보다는 밸브 frame에 대한 상대적인 frame이 들어가면??
#             3. joint torque -> 이건 밸브 강성에 따라 달라지니까 빼고.. 최대한 정보를 단순화 해서! ->그래도 있으면 좋을것 같긴한데.........ㅎ...
#         4. valve rotation angular velocity
#         밸브 종류에 따라 개별적으로 네트워크 학습한다고 생각하고... !
#         '''
#         q_unscaled = self.data.qpos[0:self.k]
#         self.obs_q = (q_unscaled - self.q_range[:, 0]) / (self.q_range[:, 1] - self.q_range[:, 0]) * (1 - (-1)) - 1
#
#         self.obs_ee = self.controller.relative_T_hand()
#         tau_unscaled = self.data.ctrl[0:self.k]
#         self.obs_tau = (tau_unscaled - self.tau_range[:, 0]) / (self.tau_range[:, 1] - self.tau_range[:, 0]) * (
#                     1 - (-1)) - 1
#
#         self.obs_omega = np.array(self.data.qvel[-2:])
#
#         obs = np.concatenate((self.obs_q, self.obs_ee, self.obs_tau, self.obs_omega), axis=0)
#         self.history_observation[1:] = self.history_observation[:-1]
#         self.history_observation[0] = obs
#         observation = self.generate_downsampled_observation(self.history_observation)
#         return observation
#
#     def _reward(self, action_rotation, action_force):
#         reward_force = 0
#         # reward_rotation = 1
#         reward_rotation = (self.door_angle - self.door_angle_pre) * 1e3
#         reward_qvel = -abs(self.data.qvel[:7]).sum() * 0.25
#         q_max = max(abs(self.obs_q))
#         if q_max > 0.9:
#             if action_force < 0:
#                 reward_force += 1
#         else:
#             if 0.1 <= abs(self.obs_omega[0]) <= 0.15:
#                 reward_force += 1
#         self.contact_detection = -1 in self.contact_list
#         if self.contact_detection:
#             idx = np.where(np.array(self.contact_list) == -1)
#             contact_force = 0.0
#             for i in idx[0]:
#                 contact_force += self.data.contact[i].dist
#             reward_rotation += np.log(-contact_force) * 0.1
#
#         # print(self.door_angle - self.door_angle_pre)
#         if self.deviation_done:
#             reward_rotation -= 100
#         elif self.bound_done:
#             reward_rotation -= 100
#             reward_force -= 100
#         elif self.time_done:
#             reward_rotation += 10
#             reward_force += 10
#
#         reward_acc = -sum(abs(action_rotation - self.action_rotation_pre))
#         # print(reward_rotation, ", ", reward_acc, ",  ", reward_rotation + reward_acc)
#         # print(abs(self.data.qvel[:7]).sum(), reward_rotation+reward_acc, reward_force)
#         return reward_rotation+reward_acc, reward_force
#
#     def _done(self):
#
#         self.contact_list = tools.detect_contact(self.data.contact, self.desired_contact_bid)
#         self.grasp_list = tools.detect_grasp(self.data.contact, self.desired_contact_finger_bid,
#                                              self.desired_contact_obj_bid)
#         self.q_operation_list = tools.detect_q_operation(self.data.qpos, self.q_range)
#
#         # self.contact_done = -1 in self.contact_list
#         self.bound_done = -1 in self.q_operation_list
#         normalized_q = self.obs_q
#         if max(abs(normalized_q)) > 0.98:
#             self.bound_done = 1
#         else:
#             self.bound_done = 0
#
#         if self.control_mode != RL_CIRCULAR_CONTROL:
#             self.deviation_done = False
#         else:
#             if len(self.grasp_list) <= 2:
#                 self.deviation_done = True
#
#         self.time_done = self.data.time - self.start_time >= self.episode_time
#         if self.time_done or self.bound_done or self.deviation_done:
#             return True
#         else:
#             return False
#
#     def _info(self):
#         info = {
#             "collision": self.contact_done,
#             "bound": self.bound_done,
#         }
#         return info
#
#     def _construct_action_space(self):
#         action_space = 2
#         action_low = -10 * np.ones(action_space)
#         action_high = 10 * np.ones(action_space)
#         rotation_action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
#
#         action_space = 1
#         action_low = -1 * np.ones(action_space)
#         action_high = 1 * np.ones(action_space)
#         force_action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
#
#         return rotation_action_space, force_action_space
#
#     def _construct_observation_space(self):
#
#         s = {
#             'q': spaces.Box(shape=(self.k, 1), low=-1, high=1, dtype=np.float32),
#             'relative_matrix': spaces.Box(shape=(16, 1), low=-np.inf, high=np.inf, dtype=np.float_),
#             'tau': spaces.Box(shape=(self.k, 1), low=-1, high=1, dtype=np.float_),
#             'omega': spaces.Box(shape=(2, 1), low=-np.inf, high=np.inf, dtype=np.float_),
#         }
#
#         observation = spaces.Dict(s)
#         feature_shape = 0
#         for _, v in s.items():
#             feature_shape += v.shape[0] * v.shape[1]
#         observation.shape = feature_shape
#         return observation
#
#     def generate_downsampled_observation(self, observation_history):
#         input_state = np.empty((self.len_hist, self.observation_space.shape))
#         j = 0
#         for i in range(0, len(observation_history), self.downsampling):
#             input_state[j] = observation_history[i]
#             j = j + 1
#         return input_state
#
#     def render(self):
#         if self.viewer is None:
#             self.viewer = viewer.launch_passive(model=self.model, data=self.data)
#         else:
#             self.viewer.sync()
#
#     def env_information(self):
#
#         latch_parents_index = mujoco.mj_name2id(self.model, BODY, "latch")
#         door = np.ones([4,4])
#         latch = np.ones([4,4])
#
#         door[3] = np.array([0,0,0,1])
#         latch[3] = np.array([0,0,0,1])
#
#         door[0:3,3] = self.data.xpos[self.door_body_idx]
#         latch[0:3,3] = self.data.xpos[self.latch_body_idx]
#         door_quat = self.model.body_quat[self.door_body_idx]
#         latch_quat = self.model.body_quat[self.latch_body_idx]
#         door[0:3, 0:3] = R.from_quat(tools.quat2xyzw(door_quat)).as_matrix()
#
#         latch_parent = R.from_quat(tools.quat2xyzw(self.model.body_quat[latch_parents_index]))
#         latch[0:3, 0:3] = (latch_parent * R.from_quat(tools.quat2xyzw(latch_quat))).as_matrix()
#
#         return latch.tolist(), door.tolist()
#
#     def save_frame_data(self, ee):
#         r = R.from_euler('xyz', ee[1][3:6], degrees=False)
#         rpyfromvalve_rot = r.inv() * R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)
#         ee_align = R.from_euler('z', 45, degrees=True)
#         rpyfromvalve = (ee_align * rpyfromvalve_rot).as_matrix()
#
#         xyzfromvalve_rot = (R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)).as_matrix()
#         xyzfromvalve_rot = np.concatenate([xyzfromvalve_rot, [[0, 0, 0]]], axis=0)
#         xyzfromvalve_rot = np.concatenate(
#             [xyzfromvalve_rot, [[self.obj_normal[0]], [self.obj_normal[1]], [self.obj_normal[2]], [1]]], axis=1)
#
#         xyzfromvalve = inv(xyzfromvalve_rot) @ np.array([[ee[1][0]], [ee[1][1]], [ee[1][2]], [1]])
#
#         if len(self.rpyfromvalve_data) == 0:
#             self.rpyfromvalve_data = rpyfromvalve.reshape(1, 3, 3)
#             self.xyzfromvalve_data = xyzfromvalve[0:3].reshape(1, 3)
#             self.gripper_data = ee[2]
#         else:
#             self.rpyfromvalve_data = np.concatenate([self.rpyfromvalve_data, [rpyfromvalve]], axis=0)
#             self.xyzfromvalve_data = np.concatenate([self.xyzfromvalve_data, [xyzfromvalve[0:3].reshape(3)]], axis=0)
#             self.gripper_data = np.concatenate([self.gripper_data, ee[2]], axis=0)
#
#     def read_file(self):
#         with open(
#                 '/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dr_heuristic.txt',
#                 'r') as f:
#             f_line = f.readline()  # 파일 한 줄 읽어오기
#             f_list = f_line.split()  # 그 줄을 list에 저장
#
#             self.dr = list(map(float, f_list))
#         with open(
#                 '/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dp_heuristic.txt',
#                 'r') as f:
#             f_line = f.readline()  # 파일 한 줄 읽어오기
#             f_list = f_line.split()  # 그 줄을 list에 저장
#
#             self.dp = list(map(float, f_list))
#
#         with open(
#                 '/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dy_heuristic.txt',
#                 'r') as f:
#             f_line = f.readline()  # 파일 한 줄 읽어오기
#             f_list = f_line.split()  # 그 줄을 list에 저장
#
#             self.dy = list(map(float, f_list))
#
#     def mujoco_xml(self):
#         if self.rendering:
#             if self.viewer is not None:
#                 self.viewer.close()
#                 self.viewer = None
#
#         del self.model
#         del self.data
#
#         # obj_list = ["handle", "valve"]
#         # o = randint(0, 1)
#         s = randint(5, 8)
#         m = randint(3, 7)
#         f = random() * 4 + 1  # 1~5
#         # f=3
#         self.friction = f
#         # s= self.episode_number +5
#         # m=3
#         f=2.5
#         # o= self.episode_number%2
#         obj = "handle"
#
#         if obj == "handle":
#             handle_xml = f'''
#                 <mujocoinclude>
#                     <compiler angle="radian" meshdir="meshes/" autolimits="true"/>
#                     <size njmax="500" nconmax="100" />
#                     <visual>
#                         <global offwidth="3024" offheight="1680" />
#                         <quality shadowsize="4096" offsamples="8" />
#                         <map force="0.1" fogend="5" />
#                     </visual>
#
#
#                     <asset>
#
#                         <mesh name="handle_base" file="objects/handle_base.STL" scale="{s} {s} {s}"/>
#                         <mesh name="handle_base0" file="objects/handle_base/handle_base000.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle_base1" file="objects/handle_base/handle_base001.obj" scale="{s} {s} {s}"/>
#
#                         <mesh name="handle" file="objects/handle.STL" scale="{s} {s} {s}"/>
#
#
#                         <mesh name="handle0" file="objects/handle2/handle000.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle1" file="objects/handle2/handle001.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle2" file="objects/handle2/handle002.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle3" file="objects/handle2/handle003.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle4" file="objects/handle2/handle004.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle5" file="objects/handle2/handle005.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle6" file="objects/handle2/handle006.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle7" file="objects/handle2/handle007.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle8" file="objects/handle2/handle008.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle9" file="objects/handle2/handle009.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle10" file="objects/handle2/handle010.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle11" file="objects/handle2/handle011.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle12" file="objects/handle2/handle012.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle13" file="objects/handle2/handle013.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle14" file="objects/handle2/handle014.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle15" file="objects/handle2/handle015.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle16" file="objects/handle2/handle016.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle17" file="objects/handle2/handle017.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle18" file="objects/handle2/handle018.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle19" file="objects/handle2/handle019.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle20" file="objects/handle2/handle020.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle21" file="objects/handle2/handle021.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle22" file="objects/handle2/handle022.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle23" file="objects/handle2/handle023.obj" scale="{s} {s} {s}"/>
#                     </asset>
#
#                     <contact>
#                         <exclude name="handle_contact" body1="handle_base" body2="handle_handle"/>
#                     </contact>
#
#                 </mujocoinclude>
#             '''
#
#             # Now you can write the XML content to a file
#             with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/assets_handle.xml',
#                       'w') as file:
#                 file.write(handle_xml)
#             handle_limit_xml = f'''
#                         <mujocoinclude>
#
#                             <body name="base_h" pos="0 0 0">
#                                 <inertial diaginertia="0 0 0" mass="0" pos="0 0 0"/>
#
#                                 <body name="handle_base" pos="0 0 0">
#                                     <inertial pos="0 0 0" mass="2.7" diaginertia="0.1 0.1 0.1" />
#                                     <geom name = "handle_base" type="mesh" rgba="1 1 1 1" mesh="handle_base" class="visual" />
#                                     <geom name = "obj_contact0" type="mesh"  mesh="handle_base0" class="collision" />
#                                     <geom type="mesh" mesh="handle_base1" class="collision"/>
#                                     <body name="handle_handle" pos="0 0 0" >
#                                         <inertial pos="0 0 0" quat="0.365653 0.605347 -0.36522 0.605365" mass="0.1" diaginertia="0.1 0.1 0.1" />
#                                         <!-- frictionloss : 벨브의 뻑뻑한 정도 결정 키울수록 돌리기 힘듦 , stiffness : 다시 원래 각도로 돌아가려는성질 : 0으로 세팅 -->
#                                         <joint name="handle_joint" pos="0 0 0" axis="0 1 0" frictionloss="{f}" damping="0" limited="false" springref="0" stiffness="0" range="-{m} {m}"/>
#                                         <geom name = "handle" type="mesh" rgba="1 0 0 1" mesh="handle" class="visual" friction="1 0.1 0.1"/>
#
#
#                                         <geom name = "handle_contact9" type="mesh"  mesh="handle9" class="collision"/><!--연결부-->
#                                         <geom name = "handle_contact13" type="mesh" mesh="handle13" class="collision" /><!--연결부-->
#                                         <geom name = "handle_contact14" type="mesh" mesh="handle14" class="collision"/><!--연결부-->
#                                         <geom name = "handle_contact17" type="mesh" mesh="handle17" class="collision"/><!--연결부-->
#                                         <geom name = "handle_contact20" type="mesh" mesh="handle20" class="collision"/><!--연결부-->
#
#                                         <geom name = "handle_contact4" type="mesh"  mesh="handle4" class="collision" /> <!--연결부십자가-->
#                                         <geom name = "handle_contact7" type="mesh"  mesh="handle7" class="collision"/> <!--연결부십자가-->
#                                         <geom name = "handle_contact18" type="mesh" mesh="handle18" class="collision"/><!--연결부십자가-->
#                                         <geom name = "handle_contact19" type="mesh" mesh="handle19" class="collision"/><!--연결부십자가-->
#
#
#                                         <geom name = "handle_contact0" type="mesh"  mesh="handle0" class="collision" />
#                                         <geom name = "handle_contact1" type="mesh"  mesh="handle1" class="collision" />
#                                         <geom name = "handle_contact2" type="mesh"  mesh="handle2" class="collision"/>
#                                         <geom name = "handle_contact3" type="mesh"  mesh="handle3" class="collision"/>
#                                         <geom name = "handle_contact5" type="mesh"  mesh="handle5" class="collision" />
#                                         <geom name = "handle_contact6" type="mesh"  mesh="handle6" class="collision"/>
#                                         <geom name = "handle_contact8" type="mesh"  mesh="handle8" class="collision" />
#                                         <geom name = "handle_contact10" type="mesh" mesh="handle10" class="collision"/>
#                                         <geom name = "handle_contact11" type="mesh" mesh="handle11" class="collision"/>
#                                         <geom name = "handle_contact12" type="mesh" mesh="handle12" class="collision" />
#                                         <geom name = "handle_contact15" type="mesh" mesh="handle15" class="collision"/>
#                                         <geom name = "handle_contact16" type="mesh" mesh="handle16" class="collision" />
#                                         <geom name = "handle_contact21" type="mesh" mesh="handle21" class="collision"/>
#                                         <geom name = "handle_contact22" type="mesh" mesh="handle22" class="collision"/>
#                                         <geom name = "handle_contact23" type="mesh" mesh="handle23" class="collision"/>
#
#                                     </body>
#                                 </body>
#                             </body>
#                         </mujocoinclude>
#                     '''
#
#             # Now you can write the XML content to a file
#             with open(
#                     '/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/mjinclude_handle.xml',
#                     'w') as file:
#                 file.write(handle_limit_xml)
#         elif obj == "valve":
#             handle_xml = f'''
#                      <mujocoinclude>
#                          <compiler angle="radian" meshdir="meshes/" autolimits="true"/>
#                          <!-- <compiler angle="radian" meshdir="meshes/"> -->
#                         <size njmax="500" nconmax="100" />
#                          <visual>
#                             <global offwidth="3024" offheight="1680" />
#                             <quality shadowsize="4096" offsamples="8" />
#                             <map force="0.1" fogend="5" />
#                         </visual>
#
#
#                       <asset>
#
#
#
#                         <mesh name="valve_base" file="objects/valve_base.STL" scale="{s} {s} {s}"/>
#                         <mesh name="valve_base0" file="objects/valve_base/valve_base000.obj" scale="{s} {s} {s}"/>
#                         <mesh name="valve_base1" file="objects/valve_base/valve_base001.obj" scale="{s} {s} {s}"/>
#                         <mesh name="valve_base2" file="objects/valve_base/valve_base002.obj" scale="{s} {s} {s}"/>
#                         <mesh name="valve_base3" file="objects/valve_base/valve_base003.obj" scale="{s} {s} {s}"/>
#                         <mesh name="valve_base4" file="objects/valve_base/valve_base004.obj" scale="{s} {s} {s}"/>
#
#                         <mesh name="valve" file="objects/valve.STL" scale="{s} {s} {s}"/>
#                         <mesh name="valve0" file="objects/valve/valve000.obj" scale="{s} {s} {s}"/>
#                         <mesh name="valve1" file="objects/valve/valve001.obj" scale="{s} {s} {s}"/>
#                         <mesh name="valve2" file="objects/valve/valve002.obj" scale="{s} {s} {s}"/>
#
#
#                       </asset>
#
#                       <contact>
#                           <exclude name="valve_contact" body1="valve_base" body2="valve_handle"/>
#                       </contact>
#
#
#                      </mujocoinclude>
#                     '''
#
#             # Now you can write the XML content to a file
#             with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/assets_valve.xml',
#                       'w') as file:
#                 file.write(handle_xml)
#
#             handle_limit_xml = f'''
#                                    <mujocoincldue>
#                                         <body name="base_v" pos="0 0 0">
#                                         <inertial diaginertia="0 0 0" mass="0" pos="0 0 0"/>
#                                         <body name="valve_base" pos="0 0 0">
#                                         <inertial pos="0 0 0" mass="2.7" diaginertia="0.1 0.1 0.1"/>
#                                         <geom name="valve_base" type="mesh" rgba="1 1 1 1" mesh="valve_base" class="visual"/>
#                                         <geom name="obj_contact7" type="mesh" mesh="valve_base0" class="collision"/>
#                                         <geom name="obj_contact8" type="mesh" mesh="valve_base1" class="collision"/>
#                                         <geom name="obj_contact9" type="mesh" mesh="valve_base2" class="collision"/>
#                                         <geom name="obj_contact10" type="mesh" mesh="valve_base3" class="collision"/>
#                                         <geom name="obj_contact11" type="mesh" mesh="valve_base4" class="collision"/>
#                                         <body name="valve_handle" pos="0 0 0">
#                                         <inertial pos="0 0 0" quat="0.365653 0.605347 -0.36522 0.605365" mass="0.1" diaginertia="0.0483771 0.0410001 0.0111013"/>
#                                         <joint name="valve_joint" pos="0 0 0" axis="0 0 1" range="-{m} {m}" frictionloss="{f}" damping="0" limited="true" springref="0" stiffness="0"/>
#                                         <geom name="valve" type="mesh" rgba="1 1 0 1" mesh="valve" class="visual"/>
#                                         <geom name="valve_contact1" type="mesh" mesh="valve0" class="collision"/>
#                                         <geom name="obj_contact13" type="mesh" mesh="valve1" class="collision"/>
#                                         <geom name="valve_contact0" type="mesh" mesh="valve2" class="collision"/>
#                                         </body>
#                                         </body>
#                                         </body>
#                                     </mujocoincldue>
#                                 '''
#
#             # Now you can write the XML content to a file
#             with open(
#                     '/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/mjinclude_valve.xml',
#                     'w') as file:
#                 file.write(handle_limit_xml)
#
#         return s, m, obj
#
# class door_env2:
#     def __init__(self) -> None:
#         self.k = 7  # for jacobian calculation
#         self.dof = 9  # all joints (include gripper joint)
#         self.model_path = "../model/scene_door.xml"
#         self.model = mujoco.MjModel.from_xml_path(self.model_path)
#         self.data = mujoco.MjData(self.model)
#         mujoco.mj_step(self.model, self.data)
#         self.controller = controller.CController(self.k)
#         self.rendering = True
#         self.train = False
#         self.env_rand = False
#         self.len_hist = 10
#         self.downsampling = 100
#
#         self.observation_space = self._construct_observation_space()
#         self.rotation_action_space, self.force_action_space = self._construct_action_space()
#
#         self.viewer = None
#         self.q_range = self.model.jnt_range[:self.k]
#         self.qdot_range = np.array([[-2.1750, 2.1750], [-2.1750, 2.1750], [-2.1750, 2.1750], [-2.1750, 2.1750],
#                                     [-2.61, 2.61], [-2.61, 2.61], [-2.61, 2.61]])
#         self.tau_range = np.array([[0, 87], [0, 87], [0, 87], [0, 87], [0, 12], [0, 12], [0, 12]])
#         self.qdot_init = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#         self.q_init = [0, np.deg2rad(-60), 0, np.deg2rad(-90), 0, np.deg2rad(90), np.deg2rad(45), 0, 0, 0, 0]
#         # self.q_init = [0, np.deg2rad(-45), 0, np.deg2rad(-135), 0, np.deg2rad(90), np.deg2rad(45), 0, 0, 0, 0]
#         self.q_reset = [0, np.deg2rad(-60), 0, np.deg2rad(-90), 0, np.deg2rad(90), np.deg2rad(45), 0.03, 0.03, 0, 0]
#         self.episode_number = -1
#         desired_contact_list = ["finger_contact0", "finger_contact1",
#                                 "finger_contact2", "finger_contact3", "finger_contact4", "finger_contact5",
#                                 "finger_contact6", "finger_contact7",
#                                 "finger_contact8", "finger_contact9", "finger0_contact", "finger1_contact",
#                                 "door_handle",]
#         desired_contact_list_finger = ["finger_contact1",
#                                        "finger_contact2", "finger_contact3", "finger_contact4",
#                                        "finger_contact6", "finger_contact7",
#                                        "finger_contact8", "finger_contact9", ]
#         desired_contact_list_obj = ["door_handle"]
#         self.desired_contact_bid = tools.name2id(self.model, GEOM, desired_contact_list)
#         self.desired_contact_finger_bid = tools.name2id(self.model, GEOM, desired_contact_list_finger)
#         self.desired_contact_obj_bid = tools.name2id(self.model, GEOM, desired_contact_list_obj)
#
#         self.history_observation = np.zeros([self.len_hist * self.downsampling, self.observation_space.shape])
#         self.goal_latch_angle = -np.pi/2
#         self.latch_radius = 0.15
#         self.goal_door_angle = +np.pi/2
#         self.door_radius = 0.73 # 문고리 회전축~문 회전축까지
#         self.door_joint_idx = mujoco.mj_name2id(self.model, JOINT, "hinge")
#         self.latch_joint_idx = mujoco.mj_name2id(self.model, JOINT, "latch")
#         self.door_body_idx = mujoco.mj_name2id(self.model, BODY, "hinge_axis")
#         self.latch_body_idx = mujoco.mj_name2id(self.model, BODY, "latch_axis")
#         self.latch_info, self.door_info = self.env_information()
#
#
#
#     def reset(self, planning_mode):
#         self.control_mode = 0
#         # env_reset = True
#         self.episode_time_elapsed = 0.0
#         # self.torque_data = []
#         self.history_observation = np.zeros((self.len_hist * self.downsampling, self.observation_space.shape))
#         self.force = 0.0
#         duration = 0
#         # while env_reset:
#         self.episode_number += 1
#
#         self.start_time = self.data.time + 100
#         self.controller.initialize(planning_mode, self.latch_info, self.door_info, [self.goal_latch_angle, self.goal_door_angle])
#
#         self.data.qpos = self.q_init
#         self.data.qvel = self.qdot_init
#
#         self.episode_time = abs(
#             MOTION_TIME_CONST * abs(self.goal_door_angle) * self.door_radius) + abs(
#             MOTION_TIME_CONST * abs(self.goal_latch_angle) * self.latch_radius)
#
#         self.controller.read(self.data.time, self.data.qpos, self.data.qvel, self.model.opt.timestep)
#         self.controller.control_mujoco()
#
#         self.contact_done = False
#         self.bound_done = False
#         self.goal_done = False
#         self.action_rotation_pre = np.zeros(2)
#
#         obs = self._observation()
#         self.max_door_angle = 0.0
#         while self.control_mode != RL_CIRCULAR_CONTROL:
#             self.control_mode = self.controller.control_mode()
#
#
#             self.controller.read(self.data.time, self.data.qpos, self.data.qvel, self.model.opt.timestep)
#             self.controller.control_mujoco()
#             self._torque, _ = self.controller.write()
#             for i in range(self.dof - 1):
#                 self.data.ctrl[i] = self._torque[i]
#                 # self.data.ctrl[i] = 0.0
#             mujoco.mj_step(self.model, self.data)
#             # self.torque_data.append(self._torque[:7])
#
#             if duration == 10:
#                 duration = 0
#                 obs = self._observation()
#             done = self._done()
#             duration += 1
#             if done:
#                 break
#
#             if self.rendering:
#                 self.render()
#
#         return obs
#
#     def step(self, action_rotation, action_force):
#
#         done = False
#         duration = 0
#         if action_force < -0.666:
#             action_force = -10
#         elif action_force < -0.333:
#             action_force = -1
#         elif action_force <= 0.333:
#             action_force = 0
#         else:
#             action_force = 1
#         self.force += action_force
#         self.force = np.clip(self.force, 0, abs(self.force))
#         while not done:
#             done = self._done()
#             self.control_mode = self.controller.control_mode()
#             self.controller.read(self.data.time, self.data.qpos, self.data.qvel, self.model.opt.timestep)
#
#             # --- RL controller input ---
#             if self.control_mode == RL_CIRCULAR_CONTROL:
#                 ddrollpitchdot_tmp = action_rotation * self.rotation_action_space.high
#                 duration += 1
#                 self.controller.put_action(ddrollpitchdot_tmp, action_force)
#             if duration == 10:
#                 break
#
#             self.controller.control_mujoco()
#             self._torque, _ = self.controller.write()
#             for i in range(self.dof - 1):
#                 self.data.ctrl[i] = self._torque[i]
#             mujoco.mj_step(self.model, self.data)
#             # self.torque_data.append(self._torque[:7])
#             # if abs(self._torque[3]) > 30:
#             #     print(self._torque[:7])
#             #     print(self._torque[:7])
#             # J = self.controller.get_jacobian()
#             # self.manipulability = tools.calc_manipulability(np.array(J))
#             if self.rendering:
#                 self.render()
#         obs = self._observation()
#         done = self._done()
#         reward_rotation, reward_force = self._reward(action_rotation, action_force)
#         info = self._info()
#
#         self.action_rotation_pre = action_rotation
#
#         return obs, reward_rotation, reward_force, done, info
#
#     def _observation(self):
#         '''
#         observation 으로 필요한 것.
#         force + rotation 가 동일한 state 를 공유 하도록 하려면??
#         1. current q
#         2. current end-effector pose - kinematics 를 잘 해석하는 네트워크가 필요하겠다..?
#         -> 이것 보다는 밸브 frame에 대한 상대적인 frame이 들어가면??
#             3. joint torque -> 이건 밸브 강성에 따라 달라지니까 빼고.. 최대한 정보를 단순화 해서! ->그래도 있으면 좋을것 같긴한데.........ㅎ...
#         4. valve rotation angular velocity
#         밸브 종류에 따라 개별적으로 네트워크 학습한다고 생각하고... !
#         '''
#         q_unscaled = self.data.qpos[0:self.k]
#         self.obs_q = (q_unscaled - self.q_range[:, 0]) / (self.q_range[:, 1] - self.q_range[:, 0]) * (1 - (-1)) - 1
#
#         self.obs_ee = self.controller.relative_T_hand()
#         tau_unscaled = self.data.ctrl[0:self.k]
#         self.obs_tau = (tau_unscaled - self.tau_range[:, 0]) / (self.tau_range[:, 1] - self.tau_range[:, 0]) * (
#                     1 - (-1)) - 1
#
#         self.obs_omega = np.array([self.data.qpos[self.door_joint_idx], self.data.qpos[self.latch_joint_idx],
#                                    self.data.qvel[self.door_joint_idx], self.data.qvel[self.latch_joint_idx]])
#
#         obs = np.concatenate((self.obs_q, self.obs_ee, self.obs_tau, self.obs_omega), axis=0)
#         self.history_observation[1:] = self.history_observation[:-1]
#         self.history_observation[0] = obs
#         observation = self.generate_downsampled_observation(self.history_observation)
#         return observation
#
#     def _reward(self, action_rotation, action_force):
#         reward_force = 0
#         reward_rotation = 1
#         q_max = max(abs(self.obs_q))
#         door_angle = self.data.qpos[self.door_joint_idx]
#         if door_angle > self.max_door_angle:
#             reward_rotation += door_angle - self.max_door_angle
#             reward_force += door_angle - self.max_door_angle
#             self.max_door_angle = door_angle
#         if q_max > 0.9:
#             if action_force < 0:
#                 reward_force += 1
#         else:
#             if 0.7 <= abs(self.obs_omega[0]) <= 0.8:
#                 reward_force += 2
#
#         self.contact_detection = -1 in self.contact_list
#         if self.contact_detection:
#             idx = np.where(np.array(self.contact_list) == -1)
#             contact_force = 0.0
#             for i in idx[0]:
#                 contact_force += self.data.contact[i].dist
#             reward_rotation += np.log(-contact_force) * 0.1
#
#         if self.deviation_done:
#             reward_rotation -= 100
#         elif self.bound_done:
#             reward_rotation -= 100
#             reward_force -= 100
#         elif self.time_done:
#             reward_rotation += 10
#             reward_force += 10
#
#         reward_acc = -sum(abs(action_rotation - self.action_rotation_pre))
#         # print(reward_rotation, ", ", reward_acc, ",  ", reward_rotation + reward_acc)
#         return reward_rotation + reward_acc, reward_force
#
#     def _done(self):
#
#         self.contact_list = tools.detect_contact(self.data.contact, self.desired_contact_bid)
#         self.grasp_list = tools.detect_grasp(self.data.contact, self.desired_contact_finger_bid,
#                                              self.desired_contact_obj_bid)
#         self.q_operation_list = tools.detect_q_operation(self.data.qpos, self.q_range)
#
#         # self.contact_done = -1 in self.contact_list
#         self.bound_done = -1 in self.q_operation_list
#         normalized_q = self.obs_q
#         if max(abs(normalized_q)) > 0.98:
#             self.bound_done = 1
#         else:
#             self.bound_done = 0
#
#         if self.control_mode != RL_CIRCULAR_CONTROL:
#             self.deviation_done = False
#         else:
#             if len(self.grasp_list) <= 2:
#                 self.deviation_done = True
#
#         self.time_done = self.data.time - self.start_time >= self.episode_time
#         if self.time_done or self.bound_done or self.deviation_done:
#             return True
#         else:
#             return False
#
#     def _info(self):
#         info = {
#             "collision": self.contact_done,
#             "bound": self.bound_done,
#         }
#         return info
#
#     def _construct_action_space(self):
#         action_space = 2
#         action_low = -10 * np.ones(action_space)
#         action_high = 10 * np.ones(action_space)
#         rotation_action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
#
#         action_space = 1
#         action_low = -1 * np.ones(action_space)
#         action_high = 1 * np.ones(action_space)
#         force_action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
#
#         return rotation_action_space, force_action_space
#
#     def _construct_observation_space(self):
#
#         s = {
#             'q': spaces.Box(shape=(self.k, 1), low=-1, high=1, dtype=np.float32),
#             'relative_matrix': spaces.Box(shape=(16, 1), low=-np.inf, high=np.inf, dtype=np.float_),
#             'tau': spaces.Box(shape=(self.k, 1), low=-1, high=1, dtype=np.float_),
#             'omega': spaces.Box(shape=(4, 1), low=-np.inf, high=np.inf, dtype=np.float_),
#         }
#
#         observation = spaces.Dict(s)
#         feature_shape = 0
#         for _, v in s.items():
#             feature_shape += v.shape[0] * v.shape[1]
#         observation.shape = feature_shape
#         return observation
#
#     def generate_downsampled_observation(self, observation_history):
#         input_state = np.empty((self.len_hist, self.observation_space.shape))
#         j = 0
#         for i in range(0, len(observation_history), self.downsampling):
#             input_state[j] = observation_history[i]
#             j = j + 1
#         return input_state
#
#     def render(self):
#         if self.viewer is None:
#             self.viewer = viewer.launch_passive(model=self.model, data=self.data)
#         else:
#             self.viewer.sync()
#
#     def env_information(self):
#
#         latch_parents_index = mujoco.mj_name2id(self.model, BODY, "latch")
#         door = np.ones([4,4])
#         latch = np.ones([4,4])
#
#         door[3] = np.array([0,0,0,1])
#         latch[3] = np.array([0,0,0,1])
#
#         door[0:3,3] = self.data.xpos[self.door_body_idx]
#         latch[0:3,3] = self.data.xpos[self.latch_body_idx]
#         door_quat = self.model.body_quat[self.door_body_idx]
#         latch_quat = self.model.body_quat[self.latch_body_idx]
#         door[0:3, 0:3] = R.from_quat(tools.quat2xyzw(door_quat)).as_matrix()
#
#         latch_parent = R.from_quat(tools.quat2xyzw(self.model.body_quat[latch_parents_index]))
#         latch[0:3, 0:3] = (latch_parent * R.from_quat(tools.quat2xyzw(latch_quat))).as_matrix()
#
#         return latch.tolist(), door.tolist()
#
#     def save_frame_data(self, ee):
#         r = R.from_euler('xyz', ee[1][3:6], degrees=False)
#         rpyfromvalve_rot = r.inv() * R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)
#         ee_align = R.from_euler('z', 45, degrees=True)
#         rpyfromvalve = (ee_align * rpyfromvalve_rot).as_matrix()
#
#         xyzfromvalve_rot = (R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)).as_matrix()
#         xyzfromvalve_rot = np.concatenate([xyzfromvalve_rot, [[0, 0, 0]]], axis=0)
#         xyzfromvalve_rot = np.concatenate(
#             [xyzfromvalve_rot, [[self.obj_normal[0]], [self.obj_normal[1]], [self.obj_normal[2]], [1]]], axis=1)
#
#         xyzfromvalve = inv(xyzfromvalve_rot) @ np.array([[ee[1][0]], [ee[1][1]], [ee[1][2]], [1]])
#
#         if len(self.rpyfromvalve_data) == 0:
#             self.rpyfromvalve_data = rpyfromvalve.reshape(1, 3, 3)
#             self.xyzfromvalve_data = xyzfromvalve[0:3].reshape(1, 3)
#             self.gripper_data = ee[2]
#         else:
#             self.rpyfromvalve_data = np.concatenate([self.rpyfromvalve_data, [rpyfromvalve]], axis=0)
#             self.xyzfromvalve_data = np.concatenate([self.xyzfromvalve_data, [xyzfromvalve[0:3].reshape(3)]], axis=0)
#             self.gripper_data = np.concatenate([self.gripper_data, ee[2]], axis=0)
#
#     def read_file(self):
#         with open(
#                 '/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dr_heuristic.txt',
#                 'r') as f:
#             f_line = f.readline()  # 파일 한 줄 읽어오기
#             f_list = f_line.split()  # 그 줄을 list에 저장
#
#             self.dr = list(map(float, f_list))
#         with open(
#                 '/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dp_heuristic.txt',
#                 'r') as f:
#             f_line = f.readline()  # 파일 한 줄 읽어오기
#             f_list = f_line.split()  # 그 줄을 list에 저장
#
#             self.dp = list(map(float, f_list))
#
#         with open(
#                 '/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dy_heuristic.txt',
#                 'r') as f:
#             f_line = f.readline()  # 파일 한 줄 읽어오기
#             f_list = f_line.split()  # 그 줄을 list에 저장
#
#             self.dy = list(map(float, f_list))
#
#     def mujoco_xml(self):
#         if self.rendering:
#             if self.viewer is not None:
#                 self.viewer.close()
#                 self.viewer = None
#
#         del self.model
#         del self.data
#
#         # obj_list = ["handle", "valve"]
#         # o = randint(0, 1)
#         s = randint(5, 8)
#         m = randint(3, 7)
#         f = random() * 4 + 1  # 1~5
#         # f=3
#         self.friction = f
#         # s= self.episode_number +5
#         # m=3
#         f=2.5
#         # o= self.episode_number%2
#         obj = "handle"
#
#         if obj == "handle":
#             handle_xml = f'''
#                 <mujocoinclude>
#                     <compiler angle="radian" meshdir="meshes/" autolimits="true"/>
#                     <size njmax="500" nconmax="100" />
#                     <visual>
#                         <global offwidth="3024" offheight="1680" />
#                         <quality shadowsize="4096" offsamples="8" />
#                         <map force="0.1" fogend="5" />
#                     </visual>
#
#
#                     <asset>
#
#                         <mesh name="handle_base" file="objects/handle_base.STL" scale="{s} {s} {s}"/>
#                         <mesh name="handle_base0" file="objects/handle_base/handle_base000.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle_base1" file="objects/handle_base/handle_base001.obj" scale="{s} {s} {s}"/>
#
#                         <mesh name="handle" file="objects/handle.STL" scale="{s} {s} {s}"/>
#
#
#                         <mesh name="handle0" file="objects/handle2/handle000.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle1" file="objects/handle2/handle001.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle2" file="objects/handle2/handle002.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle3" file="objects/handle2/handle003.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle4" file="objects/handle2/handle004.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle5" file="objects/handle2/handle005.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle6" file="objects/handle2/handle006.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle7" file="objects/handle2/handle007.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle8" file="objects/handle2/handle008.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle9" file="objects/handle2/handle009.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle10" file="objects/handle2/handle010.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle11" file="objects/handle2/handle011.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle12" file="objects/handle2/handle012.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle13" file="objects/handle2/handle013.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle14" file="objects/handle2/handle014.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle15" file="objects/handle2/handle015.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle16" file="objects/handle2/handle016.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle17" file="objects/handle2/handle017.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle18" file="objects/handle2/handle018.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle19" file="objects/handle2/handle019.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle20" file="objects/handle2/handle020.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle21" file="objects/handle2/handle021.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle22" file="objects/handle2/handle022.obj" scale="{s} {s} {s}"/>
#                         <mesh name="handle23" file="objects/handle2/handle023.obj" scale="{s} {s} {s}"/>
#                     </asset>
#
#                     <contact>
#                         <exclude name="handle_contact" body1="handle_base" body2="handle_handle"/>
#                     </contact>
#
#                 </mujocoinclude>
#             '''
#
#             # Now you can write the XML content to a file
#             with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/assets_handle.xml',
#                       'w') as file:
#                 file.write(handle_xml)
#             handle_limit_xml = f'''
#                         <mujocoinclude>
#
#                             <body name="base_h" pos="0 0 0">
#                                 <inertial diaginertia="0 0 0" mass="0" pos="0 0 0"/>
#
#                                 <body name="handle_base" pos="0 0 0">
#                                     <inertial pos="0 0 0" mass="2.7" diaginertia="0.1 0.1 0.1" />
#                                     <geom name = "handle_base" type="mesh" rgba="1 1 1 1" mesh="handle_base" class="visual" />
#                                     <geom name = "obj_contact0" type="mesh"  mesh="handle_base0" class="collision" />
#                                     <geom type="mesh" mesh="handle_base1" class="collision"/>
#                                     <body name="handle_handle" pos="0 0 0" >
#                                         <inertial pos="0 0 0" quat="0.365653 0.605347 -0.36522 0.605365" mass="0.1" diaginertia="0.1 0.1 0.1" />
#                                         <!-- frictionloss : 벨브의 뻑뻑한 정도 결정 키울수록 돌리기 힘듦 , stiffness : 다시 원래 각도로 돌아가려는성질 : 0으로 세팅 -->
#                                         <joint name="handle_joint" pos="0 0 0" axis="0 1 0" frictionloss="{f}" damping="0" limited="false" springref="0" stiffness="0" range="-{m} {m}"/>
#                                         <geom name = "handle" type="mesh" rgba="1 0 0 1" mesh="handle" class="visual" friction="1 0.1 0.1"/>
#
#
#                                         <geom name = "handle_contact9" type="mesh"  mesh="handle9" class="collision"/><!--연결부-->
#                                         <geom name = "handle_contact13" type="mesh" mesh="handle13" class="collision" /><!--연결부-->
#                                         <geom name = "handle_contact14" type="mesh" mesh="handle14" class="collision"/><!--연결부-->
#                                         <geom name = "handle_contact17" type="mesh" mesh="handle17" class="collision"/><!--연결부-->
#                                         <geom name = "handle_contact20" type="mesh" mesh="handle20" class="collision"/><!--연결부-->
#
#                                         <geom name = "handle_contact4" type="mesh"  mesh="handle4" class="collision" /> <!--연결부십자가-->
#                                         <geom name = "handle_contact7" type="mesh"  mesh="handle7" class="collision"/> <!--연결부십자가-->
#                                         <geom name = "handle_contact18" type="mesh" mesh="handle18" class="collision"/><!--연결부십자가-->
#                                         <geom name = "handle_contact19" type="mesh" mesh="handle19" class="collision"/><!--연결부십자가-->
#
#
#                                         <geom name = "handle_contact0" type="mesh"  mesh="handle0" class="collision" />
#                                         <geom name = "handle_contact1" type="mesh"  mesh="handle1" class="collision" />
#                                         <geom name = "handle_contact2" type="mesh"  mesh="handle2" class="collision"/>
#                                         <geom name = "handle_contact3" type="mesh"  mesh="handle3" class="collision"/>
#                                         <geom name = "handle_contact5" type="mesh"  mesh="handle5" class="collision" />
#                                         <geom name = "handle_contact6" type="mesh"  mesh="handle6" class="collision"/>
#                                         <geom name = "handle_contact8" type="mesh"  mesh="handle8" class="collision" />
#                                         <geom name = "handle_contact10" type="mesh" mesh="handle10" class="collision"/>
#                                         <geom name = "handle_contact11" type="mesh" mesh="handle11" class="collision"/>
#                                         <geom name = "handle_contact12" type="mesh" mesh="handle12" class="collision" />
#                                         <geom name = "handle_contact15" type="mesh" mesh="handle15" class="collision"/>
#                                         <geom name = "handle_contact16" type="mesh" mesh="handle16" class="collision" />
#                                         <geom name = "handle_contact21" type="mesh" mesh="handle21" class="collision"/>
#                                         <geom name = "handle_contact22" type="mesh" mesh="handle22" class="collision"/>
#                                         <geom name = "handle_contact23" type="mesh" mesh="handle23" class="collision"/>
#
#                                     </body>
#                                 </body>
#                             </body>
#                         </mujocoinclude>
#                     '''
#
#             # Now you can write the XML content to a file
#             with open(
#                     '/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/mjinclude_handle.xml',
#                     'w') as file:
#                 file.write(handle_limit_xml)
#         elif obj == "valve":
#             handle_xml = f'''
#                      <mujocoinclude>
#                          <compiler angle="radian" meshdir="meshes/" autolimits="true"/>
#                          <!-- <compiler angle="radian" meshdir="meshes/"> -->
#                         <size njmax="500" nconmax="100" />
#                          <visual>
#                             <global offwidth="3024" offheight="1680" />
#                             <quality shadowsize="4096" offsamples="8" />
#                             <map force="0.1" fogend="5" />
#                         </visual>
#
#
#                       <asset>
#
#
#
#                         <mesh name="valve_base" file="objects/valve_base.STL" scale="{s} {s} {s}"/>
#                         <mesh name="valve_base0" file="objects/valve_base/valve_base000.obj" scale="{s} {s} {s}"/>
#                         <mesh name="valve_base1" file="objects/valve_base/valve_base001.obj" scale="{s} {s} {s}"/>
#                         <mesh name="valve_base2" file="objects/valve_base/valve_base002.obj" scale="{s} {s} {s}"/>
#                         <mesh name="valve_base3" file="objects/valve_base/valve_base003.obj" scale="{s} {s} {s}"/>
#                         <mesh name="valve_base4" file="objects/valve_base/valve_base004.obj" scale="{s} {s} {s}"/>
#
#                         <mesh name="valve" file="objects/valve.STL" scale="{s} {s} {s}"/>
#                         <mesh name="valve0" file="objects/valve/valve000.obj" scale="{s} {s} {s}"/>
#                         <mesh name="valve1" file="objects/valve/valve001.obj" scale="{s} {s} {s}"/>
#                         <mesh name="valve2" file="objects/valve/valve002.obj" scale="{s} {s} {s}"/>
#
#
#                       </asset>
#
#                       <contact>
#                           <exclude name="valve_contact" body1="valve_base" body2="valve_handle"/>
#                       </contact>
#
#
#                      </mujocoinclude>
#                     '''
#
#             # Now you can write the XML content to a file
#             with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/assets_valve.xml',
#                       'w') as file:
#                 file.write(handle_xml)
#
#             handle_limit_xml = f'''
#                                    <mujocoincldue>
#                                         <body name="base_v" pos="0 0 0">
#                                         <inertial diaginertia="0 0 0" mass="0" pos="0 0 0"/>
#                                         <body name="valve_base" pos="0 0 0">
#                                         <inertial pos="0 0 0" mass="2.7" diaginertia="0.1 0.1 0.1"/>
#                                         <geom name="valve_base" type="mesh" rgba="1 1 1 1" mesh="valve_base" class="visual"/>
#                                         <geom name="obj_contact7" type="mesh" mesh="valve_base0" class="collision"/>
#                                         <geom name="obj_contact8" type="mesh" mesh="valve_base1" class="collision"/>
#                                         <geom name="obj_contact9" type="mesh" mesh="valve_base2" class="collision"/>
#                                         <geom name="obj_contact10" type="mesh" mesh="valve_base3" class="collision"/>
#                                         <geom name="obj_contact11" type="mesh" mesh="valve_base4" class="collision"/>
#                                         <body name="valve_handle" pos="0 0 0">
#                                         <inertial pos="0 0 0" quat="0.365653 0.605347 -0.36522 0.605365" mass="0.1" diaginertia="0.0483771 0.0410001 0.0111013"/>
#                                         <joint name="valve_joint" pos="0 0 0" axis="0 0 1" range="-{m} {m}" frictionloss="{f}" damping="0" limited="true" springref="0" stiffness="0"/>
#                                         <geom name="valve" type="mesh" rgba="1 1 0 1" mesh="valve" class="visual"/>
#                                         <geom name="valve_contact1" type="mesh" mesh="valve0" class="collision"/>
#                                         <geom name="obj_contact13" type="mesh" mesh="valve1" class="collision"/>
#                                         <geom name="valve_contact0" type="mesh" mesh="valve2" class="collision"/>
#                                         </body>
#                                         </body>
#                                         </body>
#                                     </mujocoincldue>
#                                 '''
#
#             # Now you can write the XML content to a file
#             with open(
#                     '/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/mjinclude_valve.xml',
#                     'w') as file:
#                 file.write(handle_limit_xml)
#
#         return s, m, obj

class door_env:

    def __init__(self) -> None:
        self.k = 7  # for jacobian calculation
        self.dof = 9  # all joints (include gripper joint)
        self.model_path = "../model/scene_door.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        self.controller = controller.CController(self.k)
        self.rendering = True
        self.train = False
        self.env_rand = False
        self.len_hist = 10
        self.downsampling = 100

        self.observation_space = self._construct_observation_space()
        self.rotation_action_space, self.force_action_space = self._construct_action_space()

        self.viewer = None
        self.q_range = self.model.jnt_range[:self.k]
        self.qdot_range = np.array([[-2.1750, 2.1750], [-2.1750, 2.1750], [-2.1750, 2.1750], [-2.1750, 2.1750],
                                    [-2.61, 2.61], [-2.61, 2.61], [-2.61, 2.61]])
        self.tau_range = np.array([[0, 87], [0, 87], [0, 87], [0, 87], [0, 12], [0, 12], [0, 12]])
        self.qdot_init = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.q_init = [0, np.deg2rad(-60), 0, np.deg2rad(-90), 0, np.deg2rad(90), np.deg2rad(45), 0, 0, 0, 0]
        # self.q_init = [0, np.deg2rad(-45), 0, np.deg2rad(-135), 0, np.deg2rad(90), np.deg2rad(45), 0, 0, 0, 0]
        self.q_reset = [0, np.deg2rad(-60), 0, np.deg2rad(-90), 0, np.deg2rad(90), np.deg2rad(45), 0.03, 0.03, 0, 0]
        self.episode_number = -1
        desired_contact_list = ["finger_contact0", "finger_contact1",
                                "finger_contact2", "finger_contact3", "finger_contact4", "finger_contact5",
                                "finger_contact6", "finger_contact7",
                                "finger_contact8", "finger_contact9", "finger0_contact", "finger1_contact",
                                "door_handle",]
        desired_contact_list_finger = ["finger_contact0", "finger_contact1",
                                       "finger_contact2", "finger_contact3", "finger_contact4",
                                       "finger_contact5", "finger_contact6", "finger_contact7",
                                       "finger_contact8", "finger_contact9", ]
        desired_contact_list_obj = ["door_handle"]
        self.desired_contact_bid = tools.name2id(self.model, GEOM, desired_contact_list)
        self.desired_contact_finger_bid = tools.name2id(self.model, GEOM, desired_contact_list_finger)
        self.desired_contact_obj_bid = tools.name2id(self.model, GEOM, desired_contact_list_obj)

        self.history_observation = np.zeros([self.len_hist * self.downsampling, self.observation_space.shape])
        self.goal_latch_angle = -np.pi/2
        self.latch_radius = 0.15
        self.goal_door_angle = +np.pi/2
        self.door_radius = 0.73 # 문고리 회전축~문 회전축까지
        self.door_joint_idx = mujoco.mj_name2id(self.model, JOINT, "hinge")
        self.latch_joint_idx = mujoco.mj_name2id(self.model, JOINT, "latch")
        self.door_body_idx = mujoco.mj_name2id(self.model, BODY, "hinge_axis")
        self.latch_body_idx = mujoco.mj_name2id(self.model, BODY, "latch_axis")
        self.latch_info, self.door_info = self.env_information()


    def reset(self, planning_mode):
        self.control_mode = 0
        # env_reset = True
        self.episode_time_elapsed = 0.0
        # self.torque_data = []
        self.history_observation = np.zeros((self.len_hist * self.downsampling, self.observation_space.shape))
        self.force = 0.0
        duration = 0
        # while env_reset:
        self.episode_number += 1

        self.start_time = self.data.time + 100
        self.controller.initialize(planning_mode, self.latch_info, self.door_info, [self.goal_latch_angle, self.goal_door_angle])

        self.data.qpos = self.q_init
        self.data.qvel = self.qdot_init

        self.episode_time = abs(
            MOTION_TIME_CONST * abs(self.goal_door_angle) * self.door_radius) + abs(
            MOTION_TIME_CONST * abs(self.goal_latch_angle) * self.latch_radius)

        self.controller.read(self.data.time, self.data.qpos, self.data.qvel, self.model.opt.timestep)
        self.controller.control_mujoco()

        self.contact_done = False
        self.bound_done = False
        self.goal_done = False
        self.action_rotation_pre = np.zeros(2)

        self.command_data = []

        obs = self._observation()
        while self.control_mode != RL_CIRCULAR_CONTROL:
            self.control_mode = self.controller.control_mode()


            self.controller.read(self.data.time, self.data.qpos, self.data.qvel, self.model.opt.timestep)
            self.controller.control_mujoco()
            self._torque, _ = self.controller.write()
            for i in range(self.dof - 1):
                self.data.ctrl[i] = self._torque[i]
                # self.data.ctrl[i] = 0.0
            mujoco.mj_step(self.model, self.data)
            # self.torque_data.append(self._torque[:7])

            if duration == 10:
                duration = 0
                obs = self._observation()
            done = self._done()
            duration += 1
            if done:
                break

            if self.rendering:
                self.render()
        self.door_angle = self.data.qpos[self.door_joint_idx]
        self.door_angle_pre = self.door_angle
        self.start_time = self.data.time
        return obs

    def step(self, action_rotation, action_force):

        done = False
        duration = 0
        if action_force < -0.666:
            action_force = -10
        elif action_force < -0.333:
            action_force = -1
        elif action_force <= 0.333:
            action_force = 0
        else:
            action_force = 1
        self.force += action_force
        self.force = np.clip(self.force, 0, abs(self.force))
        self.door_angle = self.data.qpos[self.door_joint_idx]
        while not done:
            done = self._done()
            self.control_mode = self.controller.control_mode()
            self.controller.read(self.data.time, self.data.qpos, self.data.qvel, self.model.opt.timestep)

            # --- RL controller input ---
            if self.control_mode == RL_CIRCULAR_CONTROL:
                ddrollpitchdot_tmp = action_rotation * self.rotation_action_space.high
                duration += 1
                self.controller.put_action(ddrollpitchdot_tmp, action_force)
            if duration == 10:
                break
            self.command_data.append(self.controller.get_commands())
            self.controller.control_mujoco()
            self._torque, _ = self.controller.write()
            for i in range(self.dof - 1):
                self.data.ctrl[i] = self._torque[i]
            mujoco.mj_step(self.model, self.data)
            # self.torque_data.append(self._torque[:7])
            # if abs(self._torque[3]) > 30:
            #     print(self._torque[:7])
            #     print(self._torque[:7])
            # J = self.controller.get_jacobian()
            # self.manipulability = tools.calc_manipulability(np.array(J))
            if self.rendering:
                self.render()
        obs = self._observation()
        done = self._done()
        reward_rotation, reward_force = self._reward(action_rotation, action_force)
        info = self._info()
        self.action_rotation_pre = action_rotation
        self.door_angle_pre = self.door_angle

        return obs, reward_rotation, reward_force, done, info

    def _observation(self):
        '''
        observation 으로 필요한 것.
        force + rotation 가 동일한 state 를 공유 하도록 하려면??
        1. current q
        2. current end-effector pose - kinematics 를 잘 해석하는 네트워크가 필요하겠다..?
        -> 이것 보다는 밸브 frame에 대한 상대적인 frame이 들어가면??
            3. joint torque -> 이건 밸브 강성에 따라 달라지니까 빼고.. 최대한 정보를 단순화 해서! ->그래도 있으면 좋을것 같긴한데.........ㅎ...
        4. valve rotation angular velocity
        밸브 종류에 따라 개별적으로 네트워크 학습한다고 생각하고... !
        '''
        q_unscaled = self.data.qpos[0:self.k]
        self.obs_q = (q_unscaled - self.q_range[:, 0]) / (self.q_range[:, 1] - self.q_range[:, 0]) * (1 - (-1)) - 1
        dq_unscaled = self.data.qvel[0:self.k]
        self.obs_dq = (dq_unscaled - self.qdot_range[:, 0]) / (self.qdot_range[:, 1] - self.qdot_range[:, 0]) * (1 - (-1)) - 1

        self.obs_ee = self.controller.relative_T_hand()
        tau_unscaled = self.data.ctrl[0:self.k]
        self.obs_tau = (tau_unscaled - self.tau_range[:, 0]) / (self.tau_range[:, 1] - self.tau_range[:, 0]) * (
                    1 - (-1)) - 1

        self.obs_omega = np.array(self.data.qvel[-2:])

        obs = np.concatenate((self.obs_q, self.obs_dq, self.obs_ee, self.obs_tau, self.obs_omega), axis=0)
        self.history_observation[1:] = self.history_observation[:-1]
        self.history_observation[0] = obs
        observation = self.generate_downsampled_observation(self.history_observation)
        return observation

    def _reward(self, action_rotation, action_force):
        reward_force = 0
        # reward_rotation = 1
        reward_rotation = (self.door_angle - self.door_angle_pre) * 1e3
        reward_qvel = -abs(self.data.qvel[:7]).sum() * 0.25
        q_max = max(abs(self.obs_q))
        if q_max > 0.9:
            if action_force < 0:
                reward_force += 2
                # reward_force += 1
        else:
            if 0.1 <= abs(self.obs_omega[0]) <= 0.15:
                reward_force += 1
        self.contact_detection = -1 in self.contact_list
        if self.contact_detection:
            idx = np.where(np.array(self.contact_list) == -1)
            contact_force = 0.0
            for i in idx[0]:
                contact_force += self.data.contact[i].dist
            reward_rotation += np.log(-contact_force) * 0.1

        # print(self.door_angle - self.door_angle_pre)
        if self.deviation_done:
            reward_rotation -= 100
        elif self.bound_done:
            reward_rotation -= 100
            reward_force -= 100
        elif self.time_done:
            reward_rotation += 10
            reward_force += 10

        reward_acc = -sum(abs(action_rotation - self.action_rotation_pre))
        # print(reward_rotation, ", ", reward_acc, ",  ", reward_rotation + reward_acc)
        # print(abs(self.data.qvel[:7]).sum(), reward_rotation+reward_acc, reward_force)
        return reward_rotation + reward_acc + reward_qvel, reward_force + reward_qvel

    def _done(self):

        self.contact_list = tools.detect_contact(self.data.contact, self.desired_contact_bid)
        self.grasp_list = tools.detect_grasp(self.data.contact, self.desired_contact_finger_bid,
                                             self.desired_contact_obj_bid)
        self.q_operation_list = tools.detect_q_operation(self.data.qpos, self.q_range)

        # self.contact_done = -1 in self.contact_list
        self.bound_done = -1 in self.q_operation_list
        normalized_q = self.obs_q
        if max(abs(normalized_q)) > 0.98:
            self.bound_done = 1
        else:
            self.bound_done = 0

        if self.control_mode != RL_CIRCULAR_CONTROL:
            self.deviation_done = False
        else:
            if len(self.grasp_list) <= 2:
                self.deviation_done = True

        self.time_done = self.data.time - self.start_time >= self.episode_time
        if self.time_done or self.bound_done or self.deviation_done:
            return True
        else:
            return False

    def _info(self):
        info = {
            "collision": self.contact_done,
            "bound": self.bound_done,
        }
        return info

    def _construct_action_space(self):
        action_space = 2
        action_low = -10 * np.ones(action_space)
        action_high = 10 * np.ones(action_space)
        rotation_action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        action_space = 1
        action_low = -1 * np.ones(action_space)
        action_high = 1 * np.ones(action_space)
        force_action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        return rotation_action_space, force_action_space

    def _construct_observation_space(self):

        s = {
            'q': spaces.Box(shape=(self.k, 1), low=-1, high=1, dtype=np.float32),
            'dq': spaces.Box(shape=(self.k, 1), low=-1, high=1, dtype=np.float32),
            'relative_matrix': spaces.Box(shape=(16, 1), low=-np.inf, high=np.inf, dtype=np.float_),
            'tau': spaces.Box(shape=(self.k, 1), low=-1, high=1, dtype=np.float_),
            'omega': spaces.Box(shape=(2, 1), low=-np.inf, high=np.inf, dtype=np.float_),
        }

        observation = spaces.Dict(s)
        feature_shape = 0
        for _, v in s.items():
            feature_shape += v.shape[0] * v.shape[1]
        observation.shape = feature_shape
        return observation

    def generate_downsampled_observation(self, observation_history):
        input_state = np.empty((self.len_hist, self.observation_space.shape))
        j = 0
        for i in range(0, len(observation_history), self.downsampling):
            input_state[j] = observation_history[i]
            j = j + 1
        return input_state

    def render(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
        else:
            self.viewer.sync()

    def env_information(self):

        latch_parents_index = mujoco.mj_name2id(self.model, BODY, "latch")
        door = np.ones([4,4])
        latch = np.ones([4,4])

        door[3] = np.array([0,0,0,1])
        latch[3] = np.array([0,0,0,1])

        door[0:3,3] = self.data.xpos[self.door_body_idx]
        latch[0:3,3] = self.data.xpos[self.latch_body_idx]
        door_quat = self.model.body_quat[self.door_body_idx]
        latch_quat = self.model.body_quat[self.latch_body_idx]
        door[0:3, 0:3] = R.from_quat(tools.quat2xyzw(door_quat)).as_matrix()

        latch_parent = R.from_quat(tools.quat2xyzw(self.model.body_quat[latch_parents_index]))
        latch[0:3, 0:3] = (latch_parent * R.from_quat(tools.quat2xyzw(latch_quat))).as_matrix()

        return latch.tolist(), door.tolist()

    def save_frame_data(self, ee):
        r = R.from_euler('xyz', ee[1][3:6], degrees=False)
        rpyfromvalve_rot = r.inv() * R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)
        ee_align = R.from_euler('z', 45, degrees=True)
        rpyfromvalve = (ee_align * rpyfromvalve_rot).as_matrix()

        xyzfromvalve_rot = (R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)).as_matrix()
        xyzfromvalve_rot = np.concatenate([xyzfromvalve_rot, [[0, 0, 0]]], axis=0)
        xyzfromvalve_rot = np.concatenate(
            [xyzfromvalve_rot, [[self.obj_normal[0]], [self.obj_normal[1]], [self.obj_normal[2]], [1]]], axis=1)

        xyzfromvalve = inv(xyzfromvalve_rot) @ np.array([[ee[1][0]], [ee[1][1]], [ee[1][2]], [1]])

        if len(self.rpyfromvalve_data) == 0:
            self.rpyfromvalve_data = rpyfromvalve.reshape(1, 3, 3)
            self.xyzfromvalve_data = xyzfromvalve[0:3].reshape(1, 3)
            self.gripper_data = ee[2]
        else:
            self.rpyfromvalve_data = np.concatenate([self.rpyfromvalve_data, [rpyfromvalve]], axis=0)
            self.xyzfromvalve_data = np.concatenate([self.xyzfromvalve_data, [xyzfromvalve[0:3].reshape(3)]], axis=0)
            self.gripper_data = np.concatenate([self.gripper_data, ee[2]], axis=0)

    def read_file(self):
        with open(
                '/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dr_heuristic.txt',
                'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dr = list(map(float, f_list))
        with open(
                '/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dp_heuristic.txt',
                'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dp = list(map(float, f_list))

        with open(
                '/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dy_heuristic.txt',
                'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dy = list(map(float, f_list))

    def mujoco_xml(self):
        if self.rendering:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None

        del self.model
        del self.data

        # obj_list = ["handle", "valve"]
        # o = randint(0, 1)
        s = randint(5, 8)
        m = randint(3, 7)
        f = random() * 4 + 1  # 1~5
        # f=3
        self.friction = f
        # s= self.episode_number +5
        # m=3
        f=2.5
        # o= self.episode_number%2
        obj = "handle"

        if obj == "handle":
            handle_xml = f'''
                <mujocoinclude>
                    <compiler angle="radian" meshdir="meshes/" autolimits="true"/>
                    <size njmax="500" nconmax="100" />
                    <visual>
                        <global offwidth="3024" offheight="1680" />
                        <quality shadowsize="4096" offsamples="8" />
                        <map force="0.1" fogend="5" />
                    </visual>


                    <asset>

                        <mesh name="handle_base" file="objects/handle_base.STL" scale="{s} {s} {s}"/>
                        <mesh name="handle_base0" file="objects/handle_base/handle_base000.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle_base1" file="objects/handle_base/handle_base001.obj" scale="{s} {s} {s}"/>

                        <mesh name="handle" file="objects/handle.STL" scale="{s} {s} {s}"/>


                        <mesh name="handle0" file="objects/handle2/handle000.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle1" file="objects/handle2/handle001.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle2" file="objects/handle2/handle002.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle3" file="objects/handle2/handle003.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle4" file="objects/handle2/handle004.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle5" file="objects/handle2/handle005.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle6" file="objects/handle2/handle006.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle7" file="objects/handle2/handle007.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle8" file="objects/handle2/handle008.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle9" file="objects/handle2/handle009.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle10" file="objects/handle2/handle010.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle11" file="objects/handle2/handle011.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle12" file="objects/handle2/handle012.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle13" file="objects/handle2/handle013.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle14" file="objects/handle2/handle014.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle15" file="objects/handle2/handle015.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle16" file="objects/handle2/handle016.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle17" file="objects/handle2/handle017.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle18" file="objects/handle2/handle018.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle19" file="objects/handle2/handle019.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle20" file="objects/handle2/handle020.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle21" file="objects/handle2/handle021.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle22" file="objects/handle2/handle022.obj" scale="{s} {s} {s}"/>
                        <mesh name="handle23" file="objects/handle2/handle023.obj" scale="{s} {s} {s}"/>
                    </asset>

                    <contact>
                        <exclude name="handle_contact" body1="handle_base" body2="handle_handle"/>
                    </contact>

                </mujocoinclude>
            '''

            # Now you can write the XML content to a file
            with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/assets_handle.xml',
                      'w') as file:
                file.write(handle_xml)
            handle_limit_xml = f'''
                        <mujocoinclude>

                            <body name="base_h" pos="0 0 0">
                                <inertial diaginertia="0 0 0" mass="0" pos="0 0 0"/>

                                <body name="handle_base" pos="0 0 0">
                                    <inertial pos="0 0 0" mass="2.7" diaginertia="0.1 0.1 0.1" />
                                    <geom name = "handle_base" type="mesh" rgba="1 1 1 1" mesh="handle_base" class="visual" />
                                    <geom name = "obj_contact0" type="mesh"  mesh="handle_base0" class="collision" />
                                    <geom type="mesh" mesh="handle_base1" class="collision"/>
                                    <body name="handle_handle" pos="0 0 0" >
                                        <inertial pos="0 0 0" quat="0.365653 0.605347 -0.36522 0.605365" mass="0.1" diaginertia="0.1 0.1 0.1" />
                                        <!-- frictionloss : 벨브의 뻑뻑한 정도 결정 키울수록 돌리기 힘듦 , stiffness : 다시 원래 각도로 돌아가려는성질 : 0으로 세팅 -->
                                        <joint name="handle_joint" pos="0 0 0" axis="0 1 0" frictionloss="{f}" damping="0" limited="false" springref="0" stiffness="0" range="-{m} {m}"/>
                                        <geom name = "handle" type="mesh" rgba="1 0 0 1" mesh="handle" class="visual" friction="1 0.1 0.1"/>


                                        <geom name = "handle_contact9" type="mesh"  mesh="handle9" class="collision"/><!--연결부-->
                                        <geom name = "handle_contact13" type="mesh" mesh="handle13" class="collision" /><!--연결부-->
                                        <geom name = "handle_contact14" type="mesh" mesh="handle14" class="collision"/><!--연결부-->
                                        <geom name = "handle_contact17" type="mesh" mesh="handle17" class="collision"/><!--연결부-->
                                        <geom name = "handle_contact20" type="mesh" mesh="handle20" class="collision"/><!--연결부-->

                                        <geom name = "handle_contact4" type="mesh"  mesh="handle4" class="collision" /> <!--연결부십자가-->
                                        <geom name = "handle_contact7" type="mesh"  mesh="handle7" class="collision"/> <!--연결부십자가-->       
                                        <geom name = "handle_contact18" type="mesh" mesh="handle18" class="collision"/><!--연결부십자가-->
                                        <geom name = "handle_contact19" type="mesh" mesh="handle19" class="collision"/><!--연결부십자가-->


                                        <geom name = "handle_contact0" type="mesh"  mesh="handle0" class="collision" />
                                        <geom name = "handle_contact1" type="mesh"  mesh="handle1" class="collision" />
                                        <geom name = "handle_contact2" type="mesh"  mesh="handle2" class="collision"/>
                                        <geom name = "handle_contact3" type="mesh"  mesh="handle3" class="collision"/>
                                        <geom name = "handle_contact5" type="mesh"  mesh="handle5" class="collision" />
                                        <geom name = "handle_contact6" type="mesh"  mesh="handle6" class="collision"/>
                                        <geom name = "handle_contact8" type="mesh"  mesh="handle8" class="collision" /> 
                                        <geom name = "handle_contact10" type="mesh" mesh="handle10" class="collision"/>
                                        <geom name = "handle_contact11" type="mesh" mesh="handle11" class="collision"/>
                                        <geom name = "handle_contact12" type="mesh" mesh="handle12" class="collision" /> 
                                        <geom name = "handle_contact15" type="mesh" mesh="handle15" class="collision"/>
                                        <geom name = "handle_contact16" type="mesh" mesh="handle16" class="collision" /> 
                                        <geom name = "handle_contact21" type="mesh" mesh="handle21" class="collision"/>
                                        <geom name = "handle_contact22" type="mesh" mesh="handle22" class="collision"/>
                                        <geom name = "handle_contact23" type="mesh" mesh="handle23" class="collision"/>

                                    </body>
                                </body>
                            </body>
                        </mujocoinclude>
                    '''

            # Now you can write the XML content to a file
            with open(
                    '/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/mjinclude_handle.xml',
                    'w') as file:
                file.write(handle_limit_xml)
        elif obj == "valve":
            handle_xml = f'''
                     <mujocoinclude>
                         <compiler angle="radian" meshdir="meshes/" autolimits="true"/>
                         <!-- <compiler angle="radian" meshdir="meshes/"> -->
                        <size njmax="500" nconmax="100" />
                         <visual>
                            <global offwidth="3024" offheight="1680" />
                            <quality shadowsize="4096" offsamples="8" />
                            <map force="0.1" fogend="5" />
                        </visual>


                      <asset>



                        <mesh name="valve_base" file="objects/valve_base.STL" scale="{s} {s} {s}"/>
                        <mesh name="valve_base0" file="objects/valve_base/valve_base000.obj" scale="{s} {s} {s}"/>
                        <mesh name="valve_base1" file="objects/valve_base/valve_base001.obj" scale="{s} {s} {s}"/>
                        <mesh name="valve_base2" file="objects/valve_base/valve_base002.obj" scale="{s} {s} {s}"/>
                        <mesh name="valve_base3" file="objects/valve_base/valve_base003.obj" scale="{s} {s} {s}"/>
                        <mesh name="valve_base4" file="objects/valve_base/valve_base004.obj" scale="{s} {s} {s}"/>

                        <mesh name="valve" file="objects/valve.STL" scale="{s} {s} {s}"/>
                        <mesh name="valve0" file="objects/valve/valve000.obj" scale="{s} {s} {s}"/>
                        <mesh name="valve1" file="objects/valve/valve001.obj" scale="{s} {s} {s}"/>
                        <mesh name="valve2" file="objects/valve/valve002.obj" scale="{s} {s} {s}"/>


                      </asset>

                      <contact>
                          <exclude name="valve_contact" body1="valve_base" body2="valve_handle"/>
                      </contact>


                     </mujocoinclude>
                    '''

            # Now you can write the XML content to a file
            with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/assets_valve.xml',
                      'w') as file:
                file.write(handle_xml)

            handle_limit_xml = f'''
                                   <mujocoincldue>
                                        <body name="base_v" pos="0 0 0">
                                        <inertial diaginertia="0 0 0" mass="0" pos="0 0 0"/>
                                        <body name="valve_base" pos="0 0 0">
                                        <inertial pos="0 0 0" mass="2.7" diaginertia="0.1 0.1 0.1"/>
                                        <geom name="valve_base" type="mesh" rgba="1 1 1 1" mesh="valve_base" class="visual"/>
                                        <geom name="obj_contact7" type="mesh" mesh="valve_base0" class="collision"/>
                                        <geom name="obj_contact8" type="mesh" mesh="valve_base1" class="collision"/>
                                        <geom name="obj_contact9" type="mesh" mesh="valve_base2" class="collision"/>
                                        <geom name="obj_contact10" type="mesh" mesh="valve_base3" class="collision"/>
                                        <geom name="obj_contact11" type="mesh" mesh="valve_base4" class="collision"/>
                                        <body name="valve_handle" pos="0 0 0">
                                        <inertial pos="0 0 0" quat="0.365653 0.605347 -0.36522 0.605365" mass="0.1" diaginertia="0.0483771 0.0410001 0.0111013"/>
                                        <joint name="valve_joint" pos="0 0 0" axis="0 0 1" range="-{m} {m}" frictionloss="{f}" damping="0" limited="true" springref="0" stiffness="0"/>
                                        <geom name="valve" type="mesh" rgba="1 1 0 1" mesh="valve" class="visual"/>
                                        <geom name="valve_contact1" type="mesh" mesh="valve0" class="collision"/>
                                        <geom name="obj_contact13" type="mesh" mesh="valve1" class="collision"/>
                                        <geom name="valve_contact0" type="mesh" mesh="valve2" class="collision"/>
                                        </body>
                                        </body>
                                        </body>
                                    </mujocoincldue> 
                                '''

            # Now you can write the XML content to a file
            with open(
                    '/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/mjinclude_valve.xml',
                    'w') as file:
                file.write(handle_limit_xml)

        return s, m, obj
