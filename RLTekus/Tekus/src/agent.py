import os
from stable_baselines3 import PPO
import torch.nn as nn

# The class used to load in a model to play on the Blue team.


class Agent:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.actor = PPO.load(os.path.join(
            cur_dir, "TekusReward1-2A-test2.zip"), custom_objects=dict(
                policy_kwargs=dict(
                    activation_fn=lambda: nn.GELU("tanh"),
                    net_arch=(dict(pi=[256, 64, 128], vf=[256, 64, 128])),
                )
        )
        )

    def act(self, state):
        return self.actor.predict(state)[0]
