import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The file used to make graphs

# train/value_loss              - Current value for the value function loss usually error between value function output and Monte-Carlo estimate (or TD(lambda) estimate)
# train/loss,                   - Current total loss value
# train/explained_variance,     - Proportion of the variability of the difference between the actual samples of the dataset and the predictions made by the model
# ev=0 => might as well have predicted zero, ev=1 => perfect prediction, ev<0 => worse than just predicting zero
# train/policy_gradient_loss    - Current value for policy gradient loss (its value holds no meaning)
# train/clip_range,             - Current value of the clipping factor for the surrogate loss of PPO
# train/n_updates,              - Number of gradient updates applied so far
# train/learning_rate,          - Learning rate
# train/approx_kl,              - Approximate mean KL divergence between old and new policy; it is an estimation of how much changes happened in the update
# train/clip_fraction,          - mean fraction of surrogate loss that was clipped (above clip_range threshold) for PPO.
# train/entropy_loss            - Mean value of the entropy loss (negative of the average policy entropy)


# convert .csv files into a pandas DataFrame, omitting the first line
data0 = pd.read_csv("./Training/Logs/TekusReward1-1A-FINAL/progress.csv")
data0 = data0.iloc[1:, :]
data1 = pd.read_csv("./Training/Logs/TekusReward1-1B-FINAL/progress.csv")
data1 = data1.iloc[1:, :]
data2 = pd.read_csv("./Training/Logs/TekusReward1-1C-FINAL/progress.csv")
data2 = data2.iloc[1:, :]
data3 = pd.read_csv("./Training/Logs/TekusReward1-1D-FINAL2/progress.csv")
data3 = data3.iloc[1:, :]
data4 = pd.read_csv("./Training/Logs/TekusReward1-1E-FINAL3/progress.csv")
data4 = data4.iloc[1:, :]
data5 = pd.read_csv("./Training/Logs/TekusReward1-1E-FINAL3cont/progress.csv")
data5 = data5.iloc[1:, :]
data6 = pd.read_csv("./Training/Logs/TekusRewardOne-OneDalone/progress.csv")
data6 = data6.iloc[1:, :]
data7 = pd.read_csv("./Training/Logs/TekusReward1-1E-ALONEcont/progress.csv")
data7 = data7.iloc[1:, :]
data8 = pd.read_csv("./Training/Logs/TekusReward1-1E-ALONEcont2/progress.csv")
data8 = data8.iloc[1:, :]

# get time step values for each file (and adjust if necessary e.g. for the CL graph)
graphing_data = data0["time/total_timesteps"].to_frame()
graphing_data["time"] = graphing_data["time/total_timesteps"]
data1_time = data1["time/total_timesteps"].to_frame() + 3_563_520
data1_time["time"] = data1_time["time/total_timesteps"]
data2_time = data2["time/total_timesteps"].to_frame() + 13_639_680
data2_time["time"] = data2_time["time/total_timesteps"]
data3_time = data3["time/total_timesteps"].to_frame() + 23_715_840
data3_time["time"] = data3_time["time/total_timesteps"]
data4_time = data4["time/total_timesteps"].to_frame() + 68_812_800
data4_time["time"] = data4_time["time/total_timesteps"]
data5_time = data5["time/total_timesteps"].to_frame() + 113_909_760
data5_time["time"] = data5_time["time/total_timesteps"]
data6_time = data6["time/total_timesteps"].to_frame()
data6_time["time"] = data6_time["time/total_timesteps"]
# data7_time = data7["time/total_timesteps"].to_frame() + 60_088_320
# data7_time["time"] = data7_time["time/total_timesteps"]
# data8_time = data8["time/total_timesteps"].to_frame() + 118_210_560
# data8_time["time"] = data8_time["time/total_timesteps"]


# get the reward for each model
graphing_data["reward0"] = data0["rollout/ep_rew_mean"]
graphing_data["reward1"] = data1["rollout/ep_rew_mean"]
graphing_data["reward2"] = data2["rollout/ep_rew_mean"]
graphing_data["reward3"] = data3["rollout/ep_rew_mean"]
graphing_data["reward4"] = data4["rollout/ep_rew_mean"]
graphing_data["reward5"] = data5["rollout/ep_rew_mean"]
graphing_data["reward6"] = data6["rollout/ep_rew_mean"]
graphing_data["reward7"] = data7["rollout/ep_rew_mean"]

# compute the rolling average, and shift left by half of the units
graphing_data["rew_rolling1.1A"] = data0["rollout/ep_rew_mean"].rolling(
    6).mean()
graphing_data["rew_rolling1.1A"] = graphing_data["rew_rolling1.1A"].shift(-3)
data1_time["rew_rolling1.1B"] = data1["rollout/ep_rew_mean"].rolling(
    6).mean()
data1_time["rew_rolling1.1B"] = data1_time["rew_rolling1.1B"].shift(-3)
data2_time["rew_rolling1.1C"] = data2["rollout/ep_rew_mean"].rolling(
    6).mean()
data2_time["rew_rolling1.1C"] = data2_time["rew_rolling1.1C"].shift(-3)
data3_time["rew_rolling1.1D"] = data3["rollout/ep_rew_mean"].rolling(
    6).mean()
data3_time["rew_rolling1.1D"] = data3_time["rew_rolling1.1D"].shift(-3)
# data4_time["rew_rolling1.2E"] = data4["rollout/ep_rew_mean"].rolling(
#     6).mean()
# data4_time["rew_rolling1.2E"] = data4_time["rew_rolling1.2E"].shift(-3)
# data5_time["rew_rolling1.2E"] = data5["rollout/ep_rew_mean"].rolling(
#     6).mean()
# data5_time["rew_rolling1.2E"] = data5_time["rew_rolling1.2E"].shift(
#     -3)
data6_time["rew_rolling1.1Dalone"] = data6["rollout/ep_rew_mean"].rolling(
    6).mean()
data6_time["rew_rolling1.1Dalone"] = data6_time["rew_rolling1.1Dalone"].shift(
    -3)
# data7_time["rew_rolling1.2Ealone"] = data7["rollout/ep_rew_mean"].rolling(
#     6).mean()
# data7_time["rew_rolling1.2Ealone"] = data7_time["rew_rolling1.2Ealone"].shift(
#     -3)
# data8_time["rew_rolling1.2Ealone"] = data8["rollout/ep_rew_mean"].rolling(
#     6).mean()
# data8_time["rew_rolling1.2Ealone"] = data8_time["rew_rolling1.2Ealone"].shift(
#     -3)

# plot the graph on the same axes
ax = graphing_data.plot(x="time", y="rew_rolling1.1A", color="b")
data1_time.plot(x="time", y="rew_rolling1.1B", ax=ax, color="y")
data2_time.plot(x="time", y="rew_rolling1.1C", ax=ax, color="g")
data3_time.plot(x="time", y="rew_rolling1.1D", ax=ax, color="purple")
# data4_time.plot(x="time", y="rew_rolling1.2E", ax=ax, color="black")
# data5_time.plot(x="time", y="rew_rolling1.2E", ax=ax, color="black")
data6_time.plot(x="time", y="rew_rolling1.1Dalone", ax=ax, color="r")
# data7_time.plot(x="time", y="rew_rolling1.2Ealone", ax=ax, color="r")
# data8_time.plot(x="time", y="rew_rolling1.2Ealone", ax=ax, color="r")
# graphing_data.plot(x="time", y="rew_rolling92", ax=ax, color="y")
# graphing_data.plot(x="time", y="rew_rolling93", ax=ax, color="g")
# graphing_data.plot(x="time", y="rew_rolling94", ax=ax, color="pink")
# graphing_data.plot(x="time", y="rew_rolling95", ax=ax, color="black")
# graphing_data.plot(x="time", y="rew_rolling96", ax=ax, color="purple")
# graphing_data.plot(x="time", y="rew_rolling97", ax=ax, color="orange")

# add any horizontal or vertical lines
# ax.axvline(x=27_500_000, color='g')
ax.axhline(y=1537, color="orange")

# set labels
ax.set_xlabel("time")
ax.set_ylabel("reward")

# limit the x or y axes, then show
plt.xlim([0, 34_652_160])
plt.show()
