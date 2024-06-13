import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# torque = np.load('./demonstration/torque_data.npy')
#
# torque = torque.transpose()
# s = len(torque)
# l = len(torque[0])
# x = np.linspace(0,l,l)
# fig = plt.plot()
# for i in range(s):
#     plt.subplot(s,1,i+1)
#     plt.plot(x, torque[i])
#     # plt.xlim([2000,3000])
# # plt.show()

def plot(data,s,axs, label=None):
    data = data.transpose()
    l = len(data[0])
    x = np.linspace(0, l, l)
    # fig = plt.plot()
    for i in range(s):
        # plt.subplot(s, 1, i + 1)
        axs[i].plot(x, data[i],'-',label=label)
        if i==0:
            axs[i].legend()
        # plt.xlim([2000,3000])
        # axs[i].ylim([-10,10])
def avg_plot(data,s,axs):
    data = data.transpose()
    x = data[0]
    # fig = plt.plot()
    for i in range(0,s):
        # plt.subplot(s, 1, i + 1)
        axs[i].plot(x, data[i])
        # plt.xlim([2000,3000])
        # axs[i].ylim([-10,10])
def exponential_moving_average(data, alpha,s):
    data = data.transpose()
    ema_data=[]
    for i in range(s):
        data_tmp = data[i]
        ema = [data_tmp[0]]  # Initialize EMA with the first value of the dataset

        for j in range(1, len(data_tmp)):
            ema.append(alpha * data_tmp[j] + (1 - alpha) * ema[j - 1])
        ema_data.append(ema)
    ema_data = np.array(ema_data).transpose()
    return np.array(ema_data)


def average(data, interval):
    s = len(data)
    data = data.transpose()
    avg_data=[]
    avg_data.extend(range(0,s, interval))
    avg_data = [avg_data]
    for i in range(len(data)):
        data_tmp = data[i]
        data_avg = []
        for j in range(0, len(data_tmp), interval):
            if j+interval < len(data_tmp):
                data_avg.append(sum(data_tmp[j:j+interval])/interval)
            else:
                data_avg.append(sum(data_tmp[j:]/len(data_tmp[j:])))
        avg_data.append(data_avg)

    avg_data = np.array(avg_data).transpose()
    return avg_data


# data_new = np.load("./log/rollpitch1/6.0/reward.npy")
# s = len(data_new[0])
# # data_ma =  exponential_moving_average(data_new, 0.2, s)
# data_avg = average(data_new, 100)
# fig, axs = plt.subplots(s, 1, figsize=(8, 6))
# plot(data_new, s, axs,"osc")
# plot(data_avg, s, axs)
# avg_plot(data_new, s, axs)
agent1 = "0609_2"
agent2 = "0610_1"
path1 = "./log/"+agent1+"/reward.npy"
path2 = "./log/"+agent2+"/reward.npy"
data1 = np.load(path1)
data2 = np.load(path2)
# data2 = average(data2, 10)
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].plot(data1[:,0], data1[:,2], label=agent1)
axs[0].plot(data2[:,0], data2[:,2], label=agent2)
plt.title("rotation reward")
axs[1].plot(data1[:,0], data1[:,3], label=agent1)
axs[1].plot(data2[:,0], data2[:,3], label=agent2)
plt.legend()
plt.title("rotation reward\n\nforce reward")

# s = len(data[0])
# fig, axs = plt.subplots(s, 1, figsize=(8, 6))
# plot(data, s, axs, "osc, y error")
#
# data = np.load("./data/torque_hybrid.npy")
# s = len(data[0])
# plot(data, s, axs, "hybrid, y error")


# data__ = np.load("./log/smooth_start5/1.0/reward.npy")
# s = len(data__[0])
# fig, axs = plt.subplots(s, 1, figsize=(8, 6))
# data_ma =  exponential_moving_average(data__, 0.2, s)
# data_avg = average(data__, 100)
# # plot(data, s, axs)
# # plot(data_ma, s, axs)
# avg_plot(data_avg, s, axs)


# data1 = np.load("./drpy_control_smooth2.npy")
# s = len(data1[0])
# fig, axs = plt.subplots(s, 1,  figsize=(8, 6))
# plot(data1, s, axs)
# # #
# data10 = np.load("./torque_smooth2.npy")
# s = len(data10[0])
# fig2, axs2 = plt.subplots(s, 1,  figsize=(8, 6))
# plot(data10, s, axs2)
#
# data = np.load("./drpy_manual.npy")
# s = len(data[0])
# # fig, axs = plt.subplots(s, 1,  figsize=(8, 6))
# plot(data, s, axs)
# fig.legend("reward_before","smooth","manual")
#
# data = np.load("./drpy_contact.npy")
# s = len(data[0])
# # fig, axs = plt.subplots(s, 1,  figsize=(8, 6))
# plot(data, s, axs)

# data__ = np.load("./data/torque_osc.npy")
#
# s = len(data__[0])
# fig, axs = plt.subplots(s-1, 1, figsize=(8, 6))
# data_ma =  exponential_moving_average(data__, 0.2, s)
# data_avg = average(data__, 100)
# plot(data__, s, axs, label = "osc")
# # plt.title("torque_osc")
# # plot(data_ma, s, axs)
# # avg_plot(data_avg, s, axs)
#
# data_ = np.load("./data/torque_hybrid.npy")
# s = len(data_[0])
# # fig, axs = plt.subplots(s, 1, figsize=(8, 6))
# data_ma =  exponential_moving_average(data_, 0.2, s)
# data_avg = average(data_, 100)
# plot(data_, s, axs, label="hybrid")
# # plot(data_ma, s, axs)
# # avg_plot(data_avg, s, axs)

# data__ = np.load("./data/compare_dynamics/x_hand_oldxml.csv")
#
# s = len(data__[0])
# fig, axs = plt.subplots(s-1, 1, figsize=(8, 6))
# data_ma =  exponential_moving_average(data__, 0.2, s)
# data_avg = average(data__, 100)
# plot(data__, s, axs, label = "osc")
# # plt.title("torque_osc")
# # plot(data_ma, s, axs)
# # avg_plot(data_avg, s, axs)
#
# data_ = np.load("./data/torque_hybrid.npy")
# s = len(data_[0])
# # fig, axs = plt.subplots(s, 1, figsize=(8, 6))
# data_ma =  exponential_moving_average(data_, 0.2, s)
# data_avg = average(data_, 100)
# plot(data_, s, axs, label="hybrid")
# # plot(data_ma, s, axs)
# # avg_plot(data_avg, s, axs)

# df = pd.read_csv("./data/compare_dynamics/q_oldxml.csv")
# data = df.to_numpy(dtype=np.float32)
# data = data[:, :7]
# s = len(data[0])
# fig, axs = plt.subplots(s, 1, figsize=(8, 6))
# plot(data, s, axs, label="q, simulation")
#
# df = pd.read_csv("./data/compare_dynamics/q_real.csv")
# data = df.to_numpy(dtype=np.float32)
# data = data[:5001, :7]
# plot(data, s, axs, label="q, real robot")

# df = pd.read_csv("./data/compare_dynamics/q_null.csv")
# data = df.to_numpy(dtype=np.float32)
# data = data[:5001, :7]
# plot(data, s, axs, label="q, simulation, add null")
#
# df = pd.read_csv("./data/compare_dynamics/q_newxml.csv")
# data = df.to_numpy(dtype=np.float32)
# data = data[:5001, :7]
# plot(data, s, axs, label="q, simulation, fix model")

# df = pd.read_csv("./data/compare_dynamics/qdot_oldxml.csv")
# data = df.to_numpy(dtype=np.float32)
# data = data[:, :7]
# s = len(data[0])
# fig, axs = plt.subplots(s, 1, figsize=(8, 6))
# plot(data, s, axs, label="qdot, simulation")
#
# df = pd.read_csv("./data/compare_dynamics/qdot_real.csv")
# data = df.to_numpy(dtype=np.float32)
# data = data[:5001, :7]
# plot(data, s, axs, label="qdot, real robot")

# df = pd.read_csv("./data/compare_dynamics/qdot_null.csv")
# data = df.to_numpy(dtype=np.float32)
# data = data[:5001, :7]
# plot(data, s, axs, label="q, simulation, add null")
#
# df = pd.read_csv("./data/compare_dynamics/qdot_newxml.csv")
# data = df.to_numpy(dtype=np.float32)
# data = data[:5001, :7]
# plot(data, s, axs, label="q, simulation, fix model")





plt.show()