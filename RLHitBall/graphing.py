import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


data_kbm = pd.read_csv("./Training/Logs/HitBallTest1-KBM/progress.csv")
data_kbm = data_kbm.iloc[1:, :]

data_def = pd.read_csv("./Training/Logs/HitBallTest1-DEFAULT/progress.csv")
data_def = data_def.iloc[1:, :]

graphing_data = data_kbm["time/total_timesteps"].to_frame()
graphing_data["time_kbm"] = graphing_data["time/total_timesteps"]
graphing_data["reward_kbm"] = data_kbm["rollout/ep_rew_mean"]
graphing_data["time_def"] = data_def["time/total_timesteps"]
graphing_data["reward_def"] = data_def["rollout/ep_rew_mean"]

graphing_data["rew_rolling_def"] = data_def["rollout/ep_rew_mean"].rolling(70).mean()
graphing_data["rew_rolling_def"] = graphing_data["rew_rolling_def"].shift(-35)
graphing_data["rew_rolling_kbm"] = data_kbm["rollout/ep_rew_mean"].rolling(70).mean()
graphing_data["rew_rolling_kbm"] = graphing_data["rew_rolling_kbm"].shift(-35)
#ax = graphing_data.plot(x="time/total_timesteps", y="rollout/ep_rew_mean", kind = "scatter")
#graphing_data.plot(x="time/total_timesteps", y="rew_mean_rolling", ax = ax, color = "r")
ax = graphing_data.plot(x="time_def", y="rew_rolling_def", color = "b")
graphing_data.plot(x="time_kbm", y="rew_rolling_kbm", ax = ax, color = "r")
ax.set_xlabel("time")
plt.ylim([-350, -50])
plt.show()