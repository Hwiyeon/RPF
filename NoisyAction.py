import attr
import numpy as np
import sys
sys.path.append('../habitat-api')
import habitat
import habitat_sim
import habitat_sim.utils
from habitat.sims.habitat_simulator.action_spaces import (
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat_sim.agent.controls import register_move_fn
import magnum as mn

"""
Register Additional Noisy Actions to Habitat-sim
for implementation of "Visual Memory for Robust Path Following"
"""

THETA = 20 # default : 10
X = 0.30 # default : 0.25
SPIN_NOISE = 1 # rad
FORWARD_NOISE = 0.5 #
FN_CLIP_RATIO = 0.2
SN_CLIP_RATIO = 0.2


@attr.s(auto_attribs=True, slots=True)
class MoveAndSpinSpec:
    forward_amount: float
    spin_amount: float
    noise : bool


def _noisy_move(
    scene_node: habitat_sim.SceneNode,
    forward_amount: float,
    spin_amount: float,
    noise : bool
):
    forward_ax = (
        np.array(scene_node.absolute_transformation().rotation_scaling())
        @ habitat_sim.geo.FRONT
    )
    forward_noise = np.random.normal(forward_amount, FORWARD_NOISE) if noise else 0
    forward = np.clip(forward_amount + forward_noise, np.maximum(forward_amount - FN_CLIP_RATIO*X,0), forward_amount + FN_CLIP_RATIO*X)
    scene_node.translate_local(forward_ax * forward)

    spin_noise = np.random.normal(0, SPIN_NOISE) * 180 / 3.141592 if noise else 0
    spin = np.clip(spin_amount + spin_noise, spin_amount - SN_CLIP_RATIO * THETA, spin_amount + SN_CLIP_RATIO * THETA)

    #print('forward : {}, spin : {}'.format(forward,spin))
    # Rotate about the +y (up) axis
    rotation_ax = habitat_sim.geo.UP
    scene_node.rotate_local(mn.Deg(spin), rotation_ax)
    # Calling normalize is needed after rotating to deal with machine precision errors
    scene_node.rotation = scene_node.rotation.normalized()



@register_move_fn(body_action=True)
class NoisyForward(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: MoveAndSpinSpec,
    ):
        #print('noisy forward : {}'.format(actuation_spec.noise))
        _noisy_move(
            scene_node,
            actuation_spec.forward_amount,
            actuation_spec.spin_amount,
            actuation_spec.noise,
        )


@register_move_fn(body_action=True)
class NoisyLeft(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: MoveAndSpinSpec,
    ):
        #print('noisy left : {}'.format(actuation_spec.noise))
        _noisy_move(
            scene_node,
            actuation_spec.forward_amount,
            actuation_spec.spin_amount,
            actuation_spec.noise,
        )

@register_move_fn(body_action=True)
class NoisyRight(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: MoveAndSpinSpec,
    ):
        #print('noisy right : {}'.format(actuation_spec.noise))
        _noisy_move(
            scene_node,
            actuation_spec.forward_amount,
            -actuation_spec.spin_amount,
            actuation_spec.noise,
        )




@habitat.registry.register_action_space_configuration
class NoNoisyMove(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[habitat.SimulatorActions.NOISY_FORWARD] = habitat_sim.ActionSpec(
            "noisy_forward",
            MoveAndSpinSpec(X, 0, noise=False)
        )
        config[habitat.SimulatorActions.NOISY_LEFT] = habitat_sim.ActionSpec(
            "noisy_left",
            MoveAndSpinSpec(0,THETA,noise=False)
        )
        config[habitat.SimulatorActions.NOISY_RIGHT] = habitat_sim.ActionSpec(
            "noisy_right",
            MoveAndSpinSpec(0,THETA,noise=False)
        )


        return config


@habitat.registry.register_action_space_configuration
class NoisyMove(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[habitat.SimulatorActions.NOISY_FORWARD] = habitat_sim.ActionSpec(
            "noisy_forward",
            MoveAndSpinSpec(X, 0, noise=True),
        )
        config[habitat.SimulatorActions.NOISY_LEFT] = habitat_sim.ActionSpec(
            "noisy_left",
            MoveAndSpinSpec(0,THETA,noise=True),
        )
        config[habitat.SimulatorActions.NOISY_RIGHT] = habitat_sim.ActionSpec(
            "noisy_right",
            MoveAndSpinSpec(0,THETA,noise=True),
        )


        return config


def main():
    habitat.SimulatorActions.extend_action_space("NOISY_LEFT")
    habitat.SimulatorActions.extend_action_space("NOISY_RIGHT")
    habitat.SimulatorActions.extend_action_space("NOISY_FORWARD")

    config = habitat.get_config(config_paths="../habitat-api/configs/tasks/pointnav.yaml")
    config.defrost()
    config.SIMULATOR.ACTION_SPACE_CONFIG = "NoNoisyMove"
    config.DATASET.DATA_PATH = "../habitat-api/" + config.DATASET.DATA_PATH
    config.SIMULATOR.SCENE = '../habitat-api/' + config.SIMULATOR.SCENE
    config.freeze()

    env = habitat.Env(config=config)
    env.reset()
    env.step(habitat.SimulatorActions.NOISY_LEFT)
    env.step(habitat.SimulatorActions.NOISY_RIGHT)
    env.step(habitat.SimulatorActions.NOISY_FORWARD)
    env.close()

    config.defrost()
    config.SIMULATOR.ACTION_SPACE_CONFIG = "NoisyMove"
    config.freeze()

    env = habitat.Env(config=config)
    env.reset()
    env.step(habitat.SimulatorActions.NOISY_LEFT)
    env.step(habitat.SimulatorActions.NOISY_RIGHT)
    env.step(habitat.SimulatorActions.NOISY_FORWARD)
    env.close()


if __name__ == "__main__":
    main()
