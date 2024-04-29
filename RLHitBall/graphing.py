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
data = pd.read_csv("./Training/Logs/TekusReward1-1Base/progress.csv")
data = data.iloc[1:, :]

# get time step values for each file
graphing_data = data["time/total_timesteps"].to_frame()
graphing_data["time"] = graphing_data["time/total_timesteps"]

# compute the rolling average, and shift left by half of the units
graphing_data["rew_rolling_def"] = data["rollout/ep_rew_mean"].rolling(
    10).mean()
graphing_data["rew_rolling_def"] = graphing_data["rew_rolling_def"].shift(-5)

# plot the graph
ax = graphing_data.plot(x="time", y="rew_rolling_def", color="b")

# set labels
ax.set_xlabel("time")
ax.set_xlabel("reward")

# limit the x or y axes, then show
# plt.ylim([600, 2000])
plt.show()
