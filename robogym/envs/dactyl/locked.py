import functools

import attr
import numpy as np

import robogym.utils.rotation as rotation
from robogym.envs.dactyl.common.cube_env import (
    CubeEnv,
    CubeSimulationInterface,
    CubeSimulationParameters,
    DactylCubeEnvConstants,
    DactylCubeEnvParameters,
)
from robogym.envs.dactyl.common.mujoco_modifiers import LockedCubeSizeModifier
from robogym.envs.dactyl.goals.locked_parallel import LockedParallelGoal
from robogym.envs.dactyl.observation.cube import (
    GoalCubeRotObservation,
    GoalIsAchievedObservation,
    GoalQposObservation,
    MujocoCubePosObservation,
    MujocoCubeRotObservation,
)
from robogym.envs.dactyl.observation.locked import GoalCubePosObservation
from robogym.envs.dactyl.observation.shadow_hand import (
    MujocoShadowhandAngleObservation,
    MujocoShadowhandRelativeFingertipsObservation,
)
from robogym.goal.goal_generator import GoalGenerator
from robogym.mujoco.mujoco_xml import MujocoXML
from robogym.observation.image import ImageObservation
from robogym.observation.mujoco import MujocoQposObservation, MujocoQvelObservation
from robogym.robot_env import ObservationMapValue as omv
from robogym.robot_env import build_nested_attr


@attr.s(auto_attribs=True)
class LockedSimulationParameters(CubeSimulationParameters):
    cube_appearance: str = "texture"


@attr.s(auto_attribs=True)
class LockedEnvParameters(DactylCubeEnvParameters):
    """ Dactyl Locked 环境的参数 - 可以在每个 episode 中更改。 """

    simulation_params: LockedSimulationParameters = build_nested_attr(
        LockedSimulationParameters
    )


@attr.s(auto_attribs=True)
class LockedEnvConstants(DactylCubeEnvConstants):
    """ Dactyl Locked 环境的常量 - 在所有 episode 中保持不变。 """

    # 在认为目标无法达到之前所经过的最大步数。
    max_steps_goal_unreachable: int = 10

    # 成功条件的阈值
    success_threshold: dict = {"cube_quat": 0.4}

    # 目标生成。
    goal_generation: str = "state"

    # 是否使用基于视觉的观察。
    vision_observations: bool = False

    # 是否使用基于视觉的目标观察。
    vision_goal: bool = False


class LockedSimulation(CubeSimulationInterface):
    """
    影手操作锁定方块的模拟。
    "锁定方块"是一个没有任何可移动部分的实心方块。
    """

    # 锁定方块模型
    CUBE_XML_PATH_PATTERN = "rubik/rubik_locked{suffix}.xml"

    @classmethod
    def _build_mujoco_cube_xml(cls, xml, cube_xml_path):
        xml.append(
            MujocoXML.parse(cube_xml_path)
            .remove_objects_by_name("annotation:outer_bound")
            .add_name_prefix("cube:")
            .set_named_objects_attr("cube:middle", tag="body", pos=[1.0, 0.87, 0.2])
            .set_named_objects_attr("cube:middle", tag="geom", density=421.0)
        )

        # 目标方块 (用于可视化)
        xml.append(
            MujocoXML.parse(cube_xml_path)
            .remove_objects_by_name("annotation:outer_bound")
            .add_name_prefix("target:")
            .set_named_objects_attr("target:middle", tag="body", pos=[1.0, 0.87, 0.2])
            # 禁用碰撞
            .set_objects_attr(tag="geom", group="2", conaffinity="0", contype="0")
        )

    @classmethod
    def _get_model_xml_path_and_params(cls, cube_appearance):
        if cube_appearance == "texture":
            cube_xml_path = cls.CUBE_XML_PATH_PATTERN.format(suffix="")
        elif cube_appearance == "material":
            cube_xml_path = cls.CUBE_XML_PATH_PATTERN.format(suffix="_material_cells")
        elif cube_appearance == "obstacles":
            cube_xml_path = cls.CUBE_XML_PATH_PATTERN.format(suffix="_with_obstacles")
        elif cube_appearance == "openai":
            cube_xml_path = cls.CUBE_XML_PATH_PATTERN.format(suffix="_openai")
        else:
            raise ValueError(f"Unrecognized cube appearance: {cube_appearance}")

        return cube_xml_path, {}

    def __init__(self, mujoco_simulation):
        super().__init__(mujoco_simulation)

        self.register_joint_group("cube_position", prefix="cube:cube_t")
        self.register_joint_group("cube_rotation", prefix="cube:cube_rot")

        self.register_joint_group("target_position", prefix="target:cube_t")
        self.register_joint_group("target_rotation", prefix="target:cube_rot")
        self.register_joint_group("target_all_joints", prefix="target:")

        self.register_joint_group("hand_angle", prefix="robot0:")


class LockedEnv(CubeEnv[LockedEnvParameters, LockedEnvConstants, LockedSimulation]):
    """
    包含影手和锁定方块（即没有移动部件，只是一个实心块）的环境。
    """

    def _default_observation_map(self):
        obs_map = {
            "cube_pos": omv({"mujoco": MujocoCubePosObservation}),
            "cube_quat": omv({"mujoco": MujocoCubeRotObservation}),
            "qpos": omv({"mujoco": MujocoQposObservation}),
            "qvel": omv({"mujoco": MujocoQvelObservation}),
            "hand_angle": omv({"mujoco": MujocoShadowhandAngleObservation}),
            "fingertip_pos": omv(
                {"mujoco": MujocoShadowhandRelativeFingertipsObservation}
            ),
            "goal_pos": omv({"goal": GoalCubePosObservation}),
            "goal_quat": omv({"goal": GoalCubeRotObservation}),
            "qpos_goal": omv({"goal": GoalQposObservation}),
            "is_goal_achieved": omv({"goal": GoalIsAchievedObservation}),
        }

        if self.constants.vision_observations:
            # 为基于视觉的策略部署添加图像观察。
            obs_map.update(
                {
                    "vision": omv(
                        {"dummy_vision": ImageObservation}, default="dummy_vision"
                    )
                }
            )

        if self.constants.vision_goal:
            obs_map.update(
                {
                    "vision_goal": omv(
                        {
                            "goal": ImageObservation,
                            "goal_render_image": ImageObservation,
                            "goal_dummy_vision": ImageObservation,
                        },
                        default="goal_dummy_vision",
                    ),
                }
            )

        return obs_map

    @classmethod
    def build_goal_generation(cls, constants, mujoco_simulation) -> GoalGenerator:
        """ 构建一个目标生成对象 """
        assert (
            constants.goal_generation == "state"
        ), "仅支持基于状态的目标生成"
        return LockedParallelGoal(mujoco_simulation)

    @classmethod
    def build_simulation(cls, constants, parameters):
        return LockedSimulation.build(
            n_substeps=constants.mujoco_substeps,
            simulation_params=parameters.simulation_params,
        )

    @classmethod
    def build_mujoco_modifiers(cls):
        modifiers = super().build_mujoco_modifiers()
        modifiers["cube_size_multiplier"] = LockedCubeSizeModifier("cube:")
        return modifiers

    ###############################################################################################
    # Internal API
    def _randomize_cube_initial_position(self):
        """ Draw a random initial position for a cube """
        # Original env had this, but I'm not really sure it's needed
        for i in range(self.constants.reset_initial_steps):
            ctrl = self.mujoco_simulation.shadow_hand.denormalize_position_control(
                self.mujoco_simulation.shadow_hand.zero_control()
            )
            self.mujoco_simulation.shadow_hand.set_position_control(ctrl)
            self.mujoco_simulation.step()

        cube_translation = (
            self._random_state.randn(3) * self.parameters.cube_position_wiggle_std
        )
        self.mujoco_simulation.add_qpos("cube_position", cube_translation)

        cube_orientation = rotation.uniform_quat(self._random_state)
        self.mujoco_simulation.set_qpos("cube_rotation", cube_orientation)

        # Need to call this after the qpos is modified
        self.mujoco_simulation.forward()

        action = self._random_state.uniform(-1.0, 1.0, self.action_space.shape[0])

        for _ in range(self.parameters.n_random_initial_steps):
            ctrl = self.mujoco_simulation.shadow_hand.denormalize_position_control(
                action
            )
            self.mujoco_simulation.shadow_hand.set_position_control(ctrl)
            self.mujoco_simulation.step()

    @classmethod
    def _get_default_wrappers(cls):
        default_wrappers = super()._get_default_wrappers()

        default_wrappers.update(
            {
                "default_observation_noise_levels": {
                    "fingertip_pos": {"uncorrelated": 0.002, "additive": 0.001},
                    "hand_angle": {"additive": 0.1, "uncorrelated": 0.1},
                    "cube_pos": {"additive": 0.005, "uncorrelated": 0.001},
                    "cube_quat": {"additive": 0.1, "uncorrelated": 0.09},
                },
                "default_no_noise_levels": {
                    "fingertip_pos": {},
                    "hand_angle": {},
                    "cube_pos": {},
                    "cube_quat": {},
                },
                "default_observation_delay_levels": {
                    "interpolators": {"cube_quat": "QuatInterpolator"},
                    "groups": {
                        # Uncomment below to enable observation delay randomization.
                        # "vision": {
                        #    "obs_names": ["cube_pos", "cube_quat"],
                        #    "mean": 3,
                        #    "std": 0.5,
                        # },
                        # "phasespace": {
                        #    "obs_names": ["fingertip_pos"],
                        #    "mean": 0.5,
                        #    "std": 0.1,
                        # }
                    },
                },
                "default_no_observation_delay_levels": {
                    "interpolators": {},
                    "groups": {},
                },
                "pre_obsnoise_randomizations": [
                    ["RandomizedActionLatency"],
                    ["RandomizedCubeSizeWrapper"],
                    ["RandomizedBodyInertiaWrapper"],
                    ["RandomizedTimestepWrapper"],
                    ["RandomizedRobotFrictionWrapper"],
                    ["RandomizedCubeFrictionWrapper"],
                    ["RandomizedGravityWrapper"],
                    ["RandomizedWindWrapper"],
                    ["RandomizedPhasespaceFingersWrapper"],
                    ["RandomizedRobotDampingWrapper"],
                    ["RandomizedRobotKpWrapper"],
                    ["RandomizedJointLimitWrapper"],
                    ["RandomizedTendonRangeWrapper"],
                ],
            }
        )

        return default_wrappers

    ###############################################################################################
    # 外部API - 用于与系统的其他部分建立通信
    @property
    def cube_type(self):
        """ 方块类型 """
        return "locked"

    ###############################################################################################
    # 完全内部的方法
    def _render_callback(self, _sim, _viewer):
        """ 设置渲染回调 """
        # 在渲染时，将目标方块（可视化）移动到预定位置并设置为目标姿态
        self.mujoco_simulation.set_qpos("target_position", np.array([0.15, 0, -0.03]))
        self.mujoco_simulation.set_qpos("target_rotation", self._goal["cube_quat"])

        self.mujoco_simulation.set_qvel("target_all_joints", 0.0)

        self.mujoco_simulation.forward()


make_simple_env = functools.partial(LockedEnv.build, apply_wrappers=False)
make_env = LockedEnv.build
