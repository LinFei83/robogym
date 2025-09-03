"""
完全垂直魔方环境 - Dactyl机器人手操作魔方的强化学习环境

这个模块实现了一个用于训练机器人手操作魔方的强化学习环境。
"完全垂直"指的是魔方的所有六个面都可以独立旋转的标准魔方。
该环境基于MuJoCo物理引擎，使用Shadow Hand机器人手模型。
"""

import functools
import typing

import attr
import numpy as np
import pycuber  # Python魔方库，用于魔方状态管理

import robogym.utils.rotation as rotation
# 导入魔方环境的基础类和接口
from robogym.envs.dactyl.common.cube_env import (
    CubeEnv,  # 魔方环境基类
    CubeSimulationInterface,  # 魔方仿真接口
    DactylCubeEnvConstants,  # Dactyl魔方环境常量
    DactylCubeEnvParameters,  # Dactyl魔方环境参数
)
from robogym.envs.dactyl.common.cube_manipulator import CubeManipulator  # 魔方操作器
from robogym.envs.dactyl.common.mujoco_modifiers import PerpendicularCubeSizeModifier  # 垂直魔方尺寸修改器

# 导入各种目标生成器
from robogym.envs.dactyl.goals.face_cube_solver import FaceCubeSolverGoal  # 面求解目标
from robogym.envs.dactyl.goals.face_curriculum import FaceCurriculumGoal  # 面课程目标
from robogym.envs.dactyl.goals.face_free import FaceFreeGoal  # 自由面目标
from robogym.envs.dactyl.goals.fixed_fair_scramble import FixedFairScrambleGoal  # 固定公平打乱目标
from robogym.envs.dactyl.goals.full_unconstrained import FullUnconstrainedGoal  # 完全无约束目标
from robogym.envs.dactyl.goals.release_cube_solver import ReleaseCubeSolverGoal  # 释放魔方求解目标
from robogym.envs.dactyl.goals.unconstrained_cube_solver import UnconstrainedCubeSolver  # 无约束魔方求解器

# 导入观察相关的类
from robogym.envs.dactyl.observation.cube import (
    GoalCubeRotObservation,  # 目标魔方旋转观察
    MujocoCubePosObservation,  # MuJoCo魔方位置观察
    MujocoCubeRotObservation,  # MuJoCo魔方旋转观察
)
from robogym.envs.dactyl.observation.full_perpendicular import (
    GoalCubePosObservation,  # 目标魔方位置观察
    GoalFaceAngleObservation,  # 目标面角度观察
    MujocoFaceAngleObservation,  # MuJoCo面角度观察
)
from robogym.envs.dactyl.observation.shadow_hand import (
    MujocoShadowhandAngleObservation,  # Shadow Hand角度观察
    MujocoShadowhandRelativeFingertipsObservation,  # Shadow Hand相对指尖位置观察
)
from robogym.goal.goal_generator import GoalGenerator  # 目标生成器基类
from robogym.mujoco.mujoco_xml import MujocoXML  # MuJoCo XML处理
from robogym.observation.mujoco import MujocoQposObservation, MujocoQvelObservation  # MuJoCo位置和速度观察
from robogym.robot_env import ObservationMapValue as omv  # 观察映射值


@attr.s(auto_attribs=True)
class FullPerpendicularEnvParameters(DactylCubeEnvParameters):
    """
    完全垂直魔方环境的参数配置类
    
    这个类定义了在每个episode中可以动态更改的环境参数。
    继承自DactylCubeEnvParameters，添加了完全垂直魔方环境特有的参数。
    使用attrs库进行自动属性生成。
    """

    # 环境初始化时执行的随机动作步数
    # 用于在每个episode开始时给环境添加一些随机性
    n_random_initial_steps: int = 10

    # 魔方尺寸的缩放倍数
    # 1.0表示标准尺寸，可以通过调整这个值来改变魔方的大小
    cube_size_multiplier: float = 1.0


@attr.s(auto_attribs=True)
class FullPerpendicularEnvConstants(DactylCubeEnvConstants):
    """
    完全垂直魔方环境的常量配置类
    
    这个类定义了环境的固定常量，这些值在环境创建时设置一次，
    在整个训练过程中保持不变。继承自DactylCubeEnvConstants。
    """

    # 每个环境步骤对应的MuJoCo物理仿真子步数
    # 更多的子步数意味着更精确的物理仿真，但计算开销更大
    mujoco_substeps: int = 10

    # 环境重置时执行的零动作步数
    # 让系统稳定下来，避免初始状态的不稳定性
    reset_initial_steps: int = 20

    # 任务成功判定的阈值字典
    # cube_quat: 魔方四元数旋转的误差阈值
    # cube_face_angle: 魔方面角度的误差阈值
    success_threshold: dict = {"cube_quat": 0.4, "cube_face_angle": 0.2}

    # 每个目标状态的最大时间步数
    # 如果在这个步数内没有完成任务，episode将结束
    max_timesteps_per_goal: int = 1600

    # 目标生成策略的类型
    # "face_free": 自由面旋转目标生成
    goal_generation: str = "face_free"

    # 面旋转的允许方向列表
    # "cw": 顺时针, "ccw": 逆时针
    goal_directions: typing.List[str] = ["cw", "ccw"]

    # 是否将目标面角度四舍五入到整数倍的90度
    # True表示只允许90度的整数倍旋转
    round_target_face: bool = True

    # 魔方重新定向与面旋转的概率
    # 0.5表示50%的概率进行魔方整体重新定向
    p_face_flip: float = 0.5

    # 初始状态下魔方打乱的步数
    # 更多的步数会产生更复杂的初始状态
    num_scramble_steps: int = 50

    # 是否在每个episode开始时打乱面的角度
    # True表示会随机设置各个面的初始角度
    scramble_face_angles: bool = True

    # 是否在每个episode开始时随机化面的角度
    # True表示会在初始化时给面角度添加随机偏移
    randomize_face_angles: bool = True


class FullPerpendicularSimulation(CubeSimulationInterface):
    """
    完全垂直魔方的物理仿真类
    
    这个类实现了Shadow Hand机器人手操作完全垂直魔方的物理仿真。
    "完全垂直魔方"指的是一个所有六个面都可以独立旋转的标准魔方，
    每个面都有完整的旋转自由度。该类基于MuJoCo物理引擎。
    
    主要功能：
    - 构建魔方和目标魔方的MuJoCo XML模型
    - 管理魔方的各种关节组（位置、旋转、驱动器等）
    - 提供魔方状态操作接口（克隆、对齐、旋转等）
    """

    @classmethod
    def _build_mujoco_cube_xml(cls, xml, cube_xml_path):
        """
        构建并修改魔方的MuJoCo XML定义
        
        这个方法负责创建两个魔方模型：
        1. 实际的物理魔方（用于仿真）
        2. 目标魔方（用于可视化目标状态）
        
        参数:
            xml: MuJoCo XML对象
            cube_xml_path: 魔方XML文件的路径
        """
        # 构建实际的物理魔方模型
        xml.append(
            MujocoXML.parse(cube_xml_path)
            .add_name_prefix("cube:")  # 添加"cube:"前缀以区分不同对象
            .set_named_objects_attr("cube:middle", tag="body", pos=[1.0, 0.87, 0.2])  # 设置魔方中心位置
            # 移除弹簧关节，简化物理模型
            .remove_objects_by_prefix(prefix="cube:cubelet:spring:", tag="joint")
        )

        # 构建目标魔方模型（仅用于可视化）
        xml.append(
            MujocoXML.parse(cube_xml_path)
            .add_name_prefix("target:")  # 添加"target:"前缀
            .set_named_objects_attr("target:middle", tag="body", pos=[1.0, 0.87, 0.2])  # 设置目标魔方位置
            # 移除弹簧关节
            .remove_objects_by_prefix(prefix="target:cubelet:spring:", tag="joint")
            # 禁用碰撞检测，使其成为纯可视化对象
            .set_objects_attr(tag="geom", group="2", conaffinity="0", contype="0")
        )

    def __init__(self, sim):
        """
        初始化完全垂直魔方仿真
        
        创建魔方操作器和目标魔方操作器，并注册所有必要的关节组。
        关节组用于批量操作相关的关节，提高操作效率。
        
        参数:
            sim: MuJoCo仿真对象
        """
        super().__init__(sim)
        
        # 创建实际魔方的操作器
        self.cube_model = CubeManipulator(prefix="cube:", sim=sim)
        # 创建目标魔方的操作器（用于可视化）
        self.target_model = CubeManipulator(prefix="target:", sim=sim)

        # 注册实际魔方的关节组
        self.register_joint_group("cube_position", prefix="cube:cube:t")  # 魔方位置关节
        self.register_joint_group("cube_rotation", prefix="cube:cube:rot")  # 魔方旋转关节
        self.register_joint_group("cube_drivers", prefix="cube:cubelet:driver:")  # 魔方驱动器关节
        self.register_joint_group("cube_cubelets", prefix="cube:cubelet:")  # 魔方小方块关节

        # 注册目标魔方的关节组
        self.register_joint_group("target_position", prefix="target:cube:t")  # 目标魔方位置关节
        self.register_joint_group("target_rotation", prefix="target:cube:rot")  # 目标魔方旋转关节
        self.register_joint_group("target_drivers", prefix="target:cubelet:driver:")  # 目标魔方驱动器关节
        self.register_joint_group("target_cubelets", prefix="target:cubelet:")  # 目标魔方小方块关节

        # 注册所有关节的组合
        self.register_joint_group("cube_all_joints", prefix="cube:")  # 实际魔方的所有关节
        self.register_joint_group("target_all_joints", prefix="target:")  # 目标魔方的所有关节

        # 注册机器人手的关节组
        self.register_joint_group("hand_angle", prefix="robot0:")  # Shadow Hand的关节角度

    def clone_target_from_cube(self):
        """
        从实际魔方状态克隆目标魔方的内部状态
        
        将实际魔方的当前状态复制到目标魔方，用于可视化当前状态
        或作为目标状态的起点。
        """
        self.set_qpos("target_cubelets", self.get_qpos("cube_cubelets"))

    def get_face_angles(self, target):
        """
        获取魔方或目标魔方的面角度
        
        返回指定魔方（实际魔方或目标魔方）的所有面的当前角度。
        面角度表示每个面相对于其初始位置的旋转角度。
        
        参数:
            target: 字符串，"cube"表示实际魔方，"target"表示目标魔方
            
        返回:
            numpy数组，包含所有面的角度值
        """
        assert target in {"cube", "target"}, f"target必须是'cube'或'target'，但得到了'{target}'"
        return self.get_qpos("{}_drivers".format(target))

    def align_target_faces(self):
        """
        将目标魔方的面对齐到垂直角度
        
        调用目标魔方操作器的软对齐功能，将所有面调整到
        最接近的垂直角度（90度的整数倍）。
        """
        self.target_model.soft_align_faces()

    def rotate_target_face(self, axis, side, angle):
        """
        按指定角度旋转目标魔方的指定面
        
        这个方法用于设置目标状态，通过旋转目标魔方的特定面
        来定义智能体需要达到的目标配置。
        
        参数:
            axis: 整数，旋转轴（0=X轴, 1=Y轴, 2=Z轴）
            side: 整数，面的一侧（0或1）
            angle: 浮点数，旋转角度（弧度）
        """
        self.target_model.rotate_face(axis, side, angle)


class FullPerpendicularEnv(
    CubeEnv[
        FullPerpendicularEnvParameters,
        FullPerpendicularEnvConstants,
        FullPerpendicularSimulation,
    ]
):
    """
    完全垂直魔方强化学习环境
    
    这是一个用于训练Shadow Hand机器人手操作魔方的强化学习环境。
    该环境旨在物理精确地模拟垂直旋转，允许魔方的所有六个面
    都可以独立旋转，实现标准魔方的完整功能。
    
    主要特点：
    - 基于MuJoCo物理引擎的精确仿真
    - 支持多种目标生成策略
    - 丰富的观察空间，包括位置、旋转、面角度等
    - 可配置的随机化和噪声
    - 支持多种成功判定标准
    
    类型参数：
        FullPerpendicularEnvParameters: 环境参数类型
        FullPerpendicularEnvConstants: 环境常量类型  
        FullPerpendicularSimulation: 仿真类型
    """

    # 目标角度的维度：六个面，每个面一个角度值
    TARGET_ANGLE_SHAPE = 6

    # 魔方各个面的几何体名称
    # 按照X轴负方向、X轴正方向、Y轴负方向、Y轴正方向、Z轴负方向、Z轴正方向的顺序
    FACE_GEOM_NAMES = [
        "cube:cubelet:neg_x",  # X轴负方向面（左面）
        "cube:cubelet:pos_x",  # X轴正方向面（右面）
        "cube:cubelet:neg_y",  # Y轴负方向面（后面）
        "cube:cubelet:pos_y",  # Y轴正方向面（前面）
        "cube:cubelet:neg_z",  # Z轴负方向面（下面）
        "cube:cubelet:pos_z",  # Z轴正方向面（上面）
    ]

    # 标准魔方操作动作列表（使用pycuber库的记号）
    # L/R: 左/右面，F/B: 前/后面，U/D: 上/下面
    # 带撇号(')表示逆时针旋转，不带撇号表示顺时针旋转
    PYCUBER_ACTIONS = ["L", "L'", "R", "R'", "F", "F'", "B", "B'", "D", "D'", "U", "U'"]

    def _default_observation_map(self):
        """
        定义默认的观察映射
        
        这个方法创建了环境的观察空间映射，定义了智能体可以观察到的
        所有信息类型及其对应的观察类。观察空间包括魔方状态、
        机器人手状态和目标状态。
        
        返回:
            字典，包含观察名称到观察映射值的映射
        """
        return {
            # 魔方相关观察
            "cube_pos": omv({"mujoco": MujocoCubePosObservation}),  # 魔方位置
            "cube_quat": omv({"mujoco": MujocoCubeRotObservation}),  # 魔方四元数旋转
            "cube_face_angle": omv({"mujoco": MujocoFaceAngleObservation}),  # 魔方面角度
            
            # 系统状态观察
            "qpos": omv({"mujoco": MujocoQposObservation}),  # 关节位置
            "qvel": omv({"mujoco": MujocoQvelObservation}),  # 关节速度
            "perp_qpos": omv({"mujoco": MujocoQposObservation}),  # 垂直关节位置（qpos的副本）
            "perp_qvel": omv({"mujoco": MujocoQvelObservation}),  # 垂直关节速度（qvel的副本）
            
            # 机器人手相关观察
            "hand_angle": omv({"mujoco": MujocoShadowhandAngleObservation}),  # Shadow Hand关节角度
            "fingertip_pos": omv(
                {"mujoco": MujocoShadowhandRelativeFingertipsObservation}  # 指尖相对位置
            ),
            
            # 目标状态观察
            "goal_pos": omv({"goal": GoalCubePosObservation}),  # 目标魔方位置
            "goal_quat": omv({"goal": GoalCubeRotObservation}),  # 目标魔方四元数旋转
            "goal_face_angle": omv({"goal": GoalFaceAngleObservation}),  # 目标魔方面角度
        }

    @classmethod
    def build_goal_generation(
        cls,
        constants: FullPerpendicularEnvConstants,
        mujoco_simulation: CubeSimulationInterface,
    ) -> GoalGenerator:
        """
        构建目标生成器对象
        
        根据配置的目标生成策略，创建相应的目标生成器。不同的目标生成器
        会产生不同类型的学习任务，从简单的面旋转到复杂的魔方求解。
        
        参数:
            constants: 环境常量配置
            mujoco_simulation: MuJoCo仿真接口
            
        返回:
            GoalGenerator: 对应的目标生成器实例
            
        异常:
            RuntimeError: 当goal_generation参数无效时抛出
        """
        if constants.goal_generation == "face_curr":
            # 面课程目标：渐进式学习，从简单到复杂的面旋转任务
            return FaceCurriculumGoal(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                goal_directions=constants.goal_directions,
                round_target_face=constants.round_target_face,
                p_face_flip=constants.p_face_flip,
            )
        elif constants.goal_generation == "face_free":
            # 自由面目标：随机选择面进行旋转，没有特定的课程结构
            return FaceFreeGoal(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                goal_directions=constants.goal_directions,
                round_target_face=constants.round_target_face,
                p_face_flip=constants.p_face_flip,
            )
        elif constants.goal_generation == "face_cube_solver":
            # 面魔方求解目标：专注于解决魔方的面层
            return FaceCubeSolverGoal(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                num_scramble_steps=constants.num_scramble_steps,
            )
        elif constants.goal_generation == "release_cube_solver":
            # 释放魔方求解目标：允许释放和重新抓取魔方的求解策略
            return ReleaseCubeSolverGoal(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                num_scramble_steps=constants.num_scramble_steps,
            )
        elif constants.goal_generation == "full_unconstrained":
            # 完全无约束目标：没有限制的完整魔方操作
            return FullUnconstrainedGoal(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                goal_directions=constants.goal_directions,
                round_target_face=constants.round_target_face,
            )
        elif constants.goal_generation == "unconstrained_cube_solver":
            # 无约束魔方求解器：完全自由的魔方求解任务
            return UnconstrainedCubeSolver(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                num_scramble_steps=constants.num_scramble_steps,
            )
        elif constants.goal_generation == "fixed_fair_scramble":
            # 固定公平打乱目标：使用固定且公平的打乱序列
            return FixedFairScrambleGoal(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                num_scramble_steps=constants.num_scramble_steps,
            )
        else:
            raise RuntimeError(
                f"无效的目标生成策略 '{constants.goal_generation}'。"
                f"支持的策略包括: face_curr, face_free, face_cube_solver, "
                f"release_cube_solver, full_unconstrained, unconstrained_cube_solver, "
                f"fixed_fair_scramble"
            )

    @classmethod
    def build_simulation(cls, constants, parameters):
        """
        构建仿真对象
        
        创建FullPerpendicularSimulation实例，配置仿真参数。
        
        参数:
            constants: 环境常量
            parameters: 环境参数
            
        返回:
            FullPerpendicularSimulation: 配置好的仿真对象
        """
        return FullPerpendicularSimulation.build(
            n_substeps=constants.mujoco_substeps,
            simulation_params=parameters.simulation_params,
        )

    @classmethod
    def build_mujoco_modifiers(cls):
        """
        构建MuJoCo修改器
        
        创建用于动态修改仿真参数的修改器对象。
        添加了魔方尺寸修改器，允许在运行时调整魔方大小。
        
        返回:
            字典: 修改器名称到修改器对象的映射
        """
        modifiers = super().build_mujoco_modifiers()
        modifiers["cube_size_multiplier"] = PerpendicularCubeSizeModifier("cube:")
        return modifiers

    ###############################################################################################
    # 内部API - 环境随机化方法
    def _scramble_cube(self):
        """
        在episode开始时随机打乱魔方
        
        使用pycuber库创建一个虚拟魔方，执行随机的魔方操作来打乱它，
        然后将打乱后的状态应用到物理仿真中的魔方上。
        """
        # 创建一个标准的已解魔方
        cube = pycuber.Cube()

        # 执行指定次数的随机操作来打乱魔方
        for i in range(self.constants.num_scramble_steps):
            action = self._random_state.choice(self.PYCUBER_ACTIONS)
            cube.perform_step(action)

        # 将打乱后的魔方状态应用到仿真中
        self.mujoco_simulation.cube_model.from_pycuber(cube)

    def _scramble_face_angles(self):
        """
        随机打乱面角度而不移动任何小方块
        
        直接设置每个面的驱动器角度到随机的90度整数倍位置，
        这样可以改变面的角度而不影响魔方的整体配置。
        """
        # 生成6个面的随机角度，每个角度都是90度的整数倍
        random_angles = self._random_state.choice([-2, -1, 0, 1, 2], size=6) * np.pi / 2
        self.mujoco_simulation.set_qpos("cube_drivers", random_angles)

    def _randomize_cube_initial_position(self):
        """
        为魔方设置随机的初始位置和状态
        
        这个方法执行完整的环境初始化随机化，包括：
        1. 稳定机器人手
        2. 随机化魔方位置和朝向
        3. 打乱魔方配置
        4. 随机化面角度
        5. 执行随机初始动作
        """
        # 首先让机器人手执行零动作以稳定系统
        for i in range(self.constants.reset_initial_steps):
            ctrl = self.mujoco_simulation.shadow_hand.denormalize_position_control(
                self.mujoco_simulation.shadow_hand.zero_control()
            )
            self.mujoco_simulation.shadow_hand.set_position_control(ctrl)
            self.mujoco_simulation.step()

        # 为魔方位置添加随机偏移
        cube_translation = (
            self._random_state.randn(3) * self.parameters.cube_position_wiggle_std
        )
        self.mujoco_simulation.add_qpos("cube_position", cube_translation)

        # 设置随机的魔方朝向
        cube_orientation = rotation.uniform_quat(self._random_state)
        self.mujoco_simulation.set_qpos("cube_rotation", cube_orientation)

        # 打乱魔方配置
        self._scramble_cube()

        # 如果启用，随机设置面角度到90度的整数倍
        if self.constants.scramble_face_angles:
            self._scramble_face_angles()

        # 如果启用，为面角度添加随机偏移
        if self.constants.randomize_face_angles:
            # 生成随机的面角度偏移（-45度到45度）
            random_face_angle = self._random_state.uniform(
                -np.pi / 4, np.pi / 4, size=2
            )
            # 随机选择一个旋转轴
            random_axis = self._random_state.randint(3)

            # 对选定轴的两个面应用随机角度偏移
            self.mujoco_simulation.cube_model.rotate_face(
                random_axis, 0, random_face_angle[0]
            )
            self.mujoco_simulation.cube_model.rotate_face(
                random_axis, 1, random_face_angle[1]
            )

        # 在修改关节位置后需要调用forward来更新物理状态
        self.mujoco_simulation.forward()

        # 生成随机动作并执行指定步数
        action = self._random_state.uniform(-1.0, 1.0, self.action_space.shape[0])

        for _ in range(self.parameters.n_random_initial_steps):
            ctrl = self.mujoco_simulation.shadow_hand.denormalize_position_control(
                action
            )
            self.mujoco_simulation.shadow_hand.set_position_control(ctrl)
            self.mujoco_simulation.step()

    ###############################################################################################
    # 外部API - 用于与系统的其他部分建立通信
    @property
    def cube_type(self):
        """
        返回魔方类型标识符
        
        返回:
            字符串: "full-perpendicular"，标识这是完全垂直魔方环境
        """
        return "full-perpendicular"

    @property
    def face_joint_names(self):
        """
        获取面关节名称列表
        
        返回魔方面关节的名称列表，去除了'cube:'前缀。
        某些包装器和工具需要使用这些名称来操作魔方的面。
        
        返回:
            列表: 不带前缀的面关节名称
        """
        return [
            # 去掉'cube:'前缀，只保留关节的实际名称
            x[5:]
            for x in self.mujoco_simulation.cube_model.joints
        ]

    ###############################################################################################
    # 内部渲染方法
    def _render_callback(self, _sim, _viewer):
        """
        渲染回调函数
        
        在每次渲染时调用，用于设置可视化效果。
        主要功能是将目标魔方移动到指定位置并设置为目标姿态，
        以便用户可以直观地看到当前目标状态。
        
        参数:
            _sim: MuJoCo仿真对象（未使用）
            _viewer: 渲染查看器对象（未使用）
        """
        # 将目标魔方（可视化用）移动到预定位置
        self.mujoco_simulation.set_qpos("target_position", np.array([0.15, 0, -0.03]))
        
        # 设置目标魔方的旋转状态为当前目标旋转
        self.mujoco_simulation.set_qpos("target_rotation", self._goal["cube_quat"])
        
        # 设置目标魔方的面角度为当前目标面角度
        self.mujoco_simulation.set_qpos("target_drivers", self._goal["cube_face_angle"])

        # 将目标魔方的所有关节速度设为零（静止状态）
        self.mujoco_simulation.set_qvel("target_all_joints", 0.0)

        # 更新物理状态以反映上述更改
        self.mujoco_simulation.forward()

    @classmethod
    def _get_default_wrappers(cls):
        """
        获取默认的环境包装器配置
        
        定义了环境的默认包装器配置，包括观察噪声、观察延迟和
        各种随机化包装器。这些包装器用于增加训练的鲁棒性，
        使模型能够更好地泛化到真实世界。
        
        返回:
            字典: 包装器配置字典
        """
        default_wrappers = super()._get_default_wrappers()

        default_wrappers.update(
            {
                # 默认观察噪声水平配置
                # 为各种观察添加噪声以模拟传感器不确定性
                "default_observation_noise_levels": {
                    "fingertip_pos": {"uncorrelated": 0.002, "additive": 0.001},  # 指尖位置噪声
                    "hand_angle": {"additive": 0.1, "uncorrelated": 0.1},  # 手部角度噪声
                    "cube_pos": {"additive": 0.005, "uncorrelated": 0.001},  # 魔方位置噪声
                    "cube_quat": {"additive": 0.1, "uncorrelated": 0.09},  # 魔方旋转噪声
                    "cube_face_angle": {"additive": 0.1, "uncorrelated": 0.1},  # 魔方面角度噪声
                },
                # 无噪声配置（用于评估或调试）
                "default_no_noise_levels": {
                    "fingertip_pos": {},
                    "hand_angle": {},
                    "cube_pos": {},
                    "cube_quat": {},
                    "cube_face_angle": {},
                },
                # 默认观察延迟配置
                # 模拟真实传感器的延迟特性
                "default_observation_delay_levels": {
                    "interpolators": {
                        "cube_quat": "QuatInterpolator",  # 四元数插值器
                        "cube_face_angle": "RadianInterpolator",  # 弧度插值器
                    },
                    "groups": {
                        # 取消注释以下配置可启用观察延迟随机化
                        # "vision": {  # 视觉系统延迟
                        #    "obs_names": ["cube_pos", "cube_quat"],
                        #    "mean": 3,  # 平均延迟3步
                        #    "std": 0.5,  # 标准差0.5步
                        # },
                        # "giiker": {  # Giiker魔方传感器延迟
                        #    "obs_names": ["cube_face_angle"],
                        #    "mean": 1,  # 平均延迟1步
                        #    "std": 0.2,  # 标准差0.2步
                        # },
                        # "phasespace": {  # 相空间追踪系统延迟
                        #    "obs_names": ["fingertip_pos"],
                        #    "mean": 0.5,  # 平均延迟0.5步
                        #    "std": 0.1,  # 标准差0.1步
                        # }
                    },
                },
                # 无观察延迟配置
                "default_no_observation_delay_levels": {
                    "interpolators": {},
                    "groups": {},
                },
                # 预观察噪声随机化包装器列表
                # 这些包装器在添加观察噪声之前应用，用于随机化物理参数
                "pre_obsnoise_randomizations": [
                    ["RandomizedActionLatency"],  # 随机化动作延迟
                    ["RandomizedPerpendicularCubeSizeWrapper"],  # 随机化垂直魔方尺寸
                    ["RandomizedBodyInertiaWrapper"],  # 随机化物体惯性
                    ["RandomizedTimestepWrapper"],  # 随机化时间步长
                    ["RandomizedRobotFrictionWrapper"],  # 随机化机器人摩擦力
                    ["RandomizedCubeFrictionWrapper"],  # 随机化魔方摩擦力
                    ["RandomizedGravityWrapper"],  # 随机化重力
                    ["RandomizedWindWrapper"],  # 随机化风力
                    ["RandomizedPhasespaceFingersWrapper"],  # 随机化相空间手指
                    ["RandomizedRobotDampingWrapper"],  # 随机化机器人阻尼
                    ["RandomizedRobotKpWrapper"],  # 随机化机器人比例增益
                    ["RandomizedFaceDampingWrapper"],  # 随机化面阻尼
                    ["RandomizedJointLimitWrapper"],  # 随机化关节限制
                    ["RandomizedTendonRangeWrapper"],  # 随机化肌腱范围
                ],
            }
        )

        return default_wrappers


# 创建简单环境的便捷函数（不应用包装器）
# 用于调试或需要直接访问原始环境的场景
make_simple_env = functools.partial(FullPerpendicularEnv.build, apply_wrappers=False)

# 创建完整环境的便捷函数（应用所有默认包装器）
# 这是推荐的环境创建方式，包含所有必要的随机化和噪声
make_env = FullPerpendicularEnv.build
