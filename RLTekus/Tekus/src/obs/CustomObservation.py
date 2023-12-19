import numpy as np
from rlgym_sim.utils import ObsBuilder
from rlgym_compat import common_values
from rlgym_sim.utils.gamestates import GameState, PlayerData, PhysicsObject


class CustomObservation(ObsBuilder):
    def reset(self, initial_state: GameState):
        pass

    def serialize(phys_obj: PhysicsObject):
        """
        Function to serialize all the values contained by this physics object into a single 1D list. This can be useful
        when constructing observations for a policy.
        :return: List containing the serialized data.
        """
        repr = []

        if phys_obj.position is not None:
            for arg in phys_obj.position:
                repr.append(arg)

        if phys_obj.quaternion is not None:
            for arg in phys_obj.quaternion:
                repr.append(arg)

        if phys_obj.linear_velocity is not None:
            for arg in phys_obj.linear_velocity:
                repr.append(arg)

        if phys_obj.angular_velocity is not None:
            for arg in phys_obj.angular_velocity:
                repr.append(arg)

        if phys_obj._euler_angles is not None:
            for arg in phys_obj._euler_angles:
                repr.append(arg)

        if phys_obj._rotation_mtx is not None:
            for arg in phys_obj._rotation_mtx.ravel():
                repr.append(arg)

        return repr

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        obs = []

        # If this observation is being built for a player on the orange team,
        # we need to invert all the physics data we use.
        inverted = player.team_num == common_values.ORANGE_TEAM

        if inverted:
            obs += CustomObservation.serialize(state.inverted_ball)
        else:
            obs += CustomObservation.serialize(state.ball)

        for player in state.players:
            if inverted:
                obs += CustomObservation.serialize(player.inverted_car_data)
            else:
                obs += CustomObservation.serialize(player.car_data)

        return np.asarray(obs, dtype=np.float32)
