**状态**: 归档 (代码按原样提供，预计不会有更新)

# Robogym

robogym 是一个模拟框架，它使用 [OpenAI gym](https://gym.openai.com/) 和 [MuJoCo 物理模拟器](http://mujoco.org/)，并提供了多种适用于不同场景下机器人学习的机器人环境。

<p align="center">
    <img src="docs/assets/dactyl/dactyl_full_perpendicular.png" height="300"/> &nbsp; <img src="docs/assets/rearrange.png" height="300"/>
</p>

## 支持的平台

该软件包已在 Mac OS Mojave、Catalina 和 Ubuntu 16.04 LTS 上进行了测试，并且可能适用于大多数最新的 Mac 和 Linux 操作系统。

需要 **Python 3.7.4 或更高版本**。

## 安装

1.  按照 `mujoco-py` 软件包中的[说明](https://github.com/openai/mujoco-py#install-mujoco)安装 MuJoCo。

2.  要检出代码并通过 `pip install` 安装，请运行：

    ```bash
    git clone git@github.com:openai/robogym.git
    cd robogym
    pip install -e .
    ```

    或者你可以直接通过以下方式安装：

    ```bash
    pip install git+https://github.com/openai/robogym.git
    ```

## 引用

请使用下面的 BibTeX 条目来引用此框架：

```
@misc{robogym2020,
  author={OpenAI},
  title={{Robogym}},
  year={2020},
  howpublished="\url{https://github.com/openai/robogym}",
}
```

# 用法

## 可视化环境

你可以使用 ```robogym/scripts/examine.py``` 来可视化环境并与之交互。

例如，以下脚本可视化了 `dactyl/locked.py` 环境。

```bash
python robogym/scripts/examine.py robogym/envs/dactyl/locked.py constants='@{"randomize": True}'
```

请注意，`constants='@{"randomize": True}` 是一个为环境设置常量的参数。

同样，你也可以设置环境的参数。下面显示了可视化具有 5 个物体的块重排环境的命令。

```bash
python robogym/scripts/examine.py robogym/envs/rearrange/blocks.py parameters='@{"simulation_params": {"num_objects": 5}}'
```

我们通过 `--teleoperate` 选项支持重排环境的遥操作，该选项允许用户通过键盘控制机器人与环境进行交互。
以下是遥操作的示例命令。

```bash
python robogym/scripts/examine.py robogym/envs/rearrange/blocks.py parameters='@{"simulation_params": {"num_objects": 5}}' --teleoperate
```

通过 `jsonnet` 配置指定的保持环境也可以使用此机制进行可视化和遥操作，如下所示

```bash
python robogym/scripts/examine.py robogym/envs/rearrange/holdouts/configs/rainbow.jsonnet --teleoperate
```

## 创建 Python 环境

这些环境扩展了 OpenAI gym，并支持 gym 提供的强化学习接口，包括 `step`、`reset`、`render` 和 `observe` 方法。

所有环境实现都在 `robogym.envs` 模块下，可以通过调用 `make_env` 函数来实例化。例如，以下代码片段创建了一个默认的锁定立方体环境：

```python
from robogym.envs.dactyl.locked import make_env
env = make_env()
```

有关如何自定义环境的详细信息，请参阅[自定义机器人环境](#customizing-robotics-environments)部分。

# 环境

所有环境类都是 `robogym.robot_env.RobotEnv` 的子类。`RobotEnv.build` 类方法是构造环境对象的主要入口点，在每个环境中由 `make_env` 指向。自定义参数和常量应由 `RobotEnvParameters` 和 `RobotEnvConstants` 的子类定义。

物理和模拟器设置封装在 `robogym.mujoco.simulation_interface.SimulationInterface` 中。`SimulationInterface` 的一个实例与 `RobotEnv` 的一个实例之间存在一对一的映射。

每个环境都包含一个可通过 `env.robot` 访问的 `robot` 对象，该对象实现了 [`RobotInterface`](robogym/robot/robot_interface.py)。

## 训练/测试环境

### Dactyl 环境

Dactyl 环境利用具有 20 个驱动自由度的 Shadow Robot 手部机器人模拟来执行手内操作任务。以下是此类别中提供的环境的完整列表：

|图片|名称|描述|
|:---|:---|:---|
|<img src="docs/assets/dactyl/dactyl_locked.png" width="200"/>|dactyl/locked.py| 操作一个没有内部自由度的锁定立方体以匹配目标姿势|
|<img src="docs/assets/dactyl/dactyl_face_perpendicular.png" width="200"/>|dactyl/face_perpendicular.py|操作一个具有 2 个内部自由度的魔方以匹配目标姿势和面部角度|
|<img src="docs/assets/dactyl/dactyl_full_perpendicular.png" width="200"/>|dactyl/full_perpendicular.py|操作一个具有完整的 6 个内部自由度的魔方以匹配目标姿势和面部角度|
|<img src="docs/assets/dactyl/dactyl_reach.png" width="200"/>|dactyl/reach.py|指尖目标位置的到达任务|

### 重排环境

这些环境基于配备有 RobotIQ 2f-85 夹持器的 UR16e 机器人，该机器人能够在桌面环境中重新排列各种物体分布。支持多种不同类型的机器人控制模式，详见[此处](robogym/robot/README.md)。
<p align="center">
<img src="docs/assets/rearrange.png" width="400"/>
</p>

提供了各种目标生成器，以支持在给定的物体分布上指定诸如 `stack`、`pick-and-place`、`reach` 和 `rearrange` 之类的不同任务。
所有重排环境及其配置的列表在此[文档](./docs/list_rearrange_env.md)中进行了描述。

以下是此类别中支持的物体分布列表：

|图片|名称|描述|
|:---|:---|:---|
|<img src="docs/assets/blocks.png" width="200"/>|rearrange/blocks.py|采样不同颜色的块|
|<img src="docs/assets/ycb.png" width="200"/>|rearrange/ycb.py|从 [YCB](https://www.ycbbenchmarks.com/) 对象中采样|
|<img src="docs/assets/composer.png" width="200"/>|rearrange/composer.py|采样由随机网格组成的对象，这些网格要么是基本几何形状，要么是随机凸网格（分解的 YCB 对象）|
|<img src="docs/assets/mixture.png" width="200"/>|rearrange/mixture.py|从网格对象分布的混合中生成对象（支持 ycb/geom 网格数据集）|

对于重排环境，我们还提供了各种通常用于评估目的的保持任务。各种保持环境的目标状态可以在下面的图像网格中看到。
<p align="center">
<img src="docs/assets/all_holdouts.png" width="800"/>
</p>

## 自定义机器人环境

大多数机器人环境都支持通过向 `make_env` 提供额外的 `constant` 参数来进行自定义，你可以通过查看 `<EnvName>Constants` 类的定义来了解每个环境支持哪些常量，该类通常与 `make_env` 位于同一文件中。
一些通常支持的常量是：

- `randomize`: 如果为 true，将对物理、动作和观察应用一些随机化。
- `mujoco_substeps`: mujoco 模拟的每步子步数，可用于平衡模拟精度和训练速度。
- `max_timesteps_per_goal`: 在超时前实现每个目标所允许的最大时间步数。

同样，还有 `parameters` 参数，可以与 `constants` 一起自定义。
你可以通过查看 `<EnvName>Parameters` 的定义来了解每个环境支持哪些参数。

以下是我们用于训练大多数机器人环境的默认设置：

```python
env = make_env(
    constants={
        'randomize': True,
        'mujoco_substeps': 10,
        'max_timesteps_per_goal': 400
    },
    parameters={
        'n_random_initial_steps': 10,
    }
)
```

## 环境随机化接口

Robogym 提供了一种在训练期间干预环境参数的方法，以支持域随机化和课程学习。

下面显示了一个干预块（重排）环境对象数量的示例。
你可以使用此接口来定义对象数量的课程：

```python
from robogym.envs.rearrange.blocks import make_env

# 创建一个具有默认对象数量的环境：5
env = make_env(
    parameters={
        'simulation_params': {
            'num_objects': 5,
            'max_num_objects': 8,
        }
    }
)

# 获取对象数量参数接口
param = env.unwrapped.randomization.get_parameter("parameters:num_objects")

# 为下一集设置 num_objects: 3
param.set_value(3)

# 重置以随机生成一个 `num_objects: 3` 的环境
obs = env.reset()
```

有关更多详细信息，请参阅["环境随机化接口"](docs/env_param_interface.md)的文档。

## 创建新的重排环境

我们提供了一套工具来帮助通过遥操作创建自定义的重排环境。

有关更多详细信息，请参阅["构建新的重排环境"](docs/build_new_rearrange_envs.md)的文档。
