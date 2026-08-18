# UFACTORY xArm7 · LeRobot（GELLO / 手动拖拽）

> [English Version](README.md)

UFACTORY xArm 与 [LeRobot](https://github.com/huggingface/lerobot) 框架的集成项目，专注于两种数据采集方式：

- **GELLO** — 使用 Dynamixel 示教臂的关节空间遥操作
- **手动拖拽** — 在 xArm 示教模式下自由拖动机械臂录制演示

采集的数据以标准 LeRobot 数据集格式保存，可用于模仿学习训练（ACT / Diffusion Policy 等）和实时策略推理。

## 功能特性

- 🤖 UFACTORY xArm7 控制
- 🎮 GELLO 关节空间遥操作（Dynamixel 示教臂）
- ✋ xArm 示教模式手动拖拽采集
- 📷 Intel RealSense 相机观测（D435 / D435i）
- 📊 兼容 LeRobot 格式的数据集录制与管理
- 🧠 模仿学习训练与策略推理
- ▶️ 手动演示数据的 episode 回放

## 环境要求

- Ubuntu 22.04 / 24.04
- Python >= 3.10
- CUDA >= 12.0（GPU 训练推荐）
- UFACTORY xArm7 及控制器
- GELLO 示教臂（FTDI USB 串口）
- Intel RealSense D435 / D435i（需要相机观测时）

## 安装

```bash
git clone https://git.weiyantech.cn/wangshuxun/Xarm-DataCollection.git lerobot_xarm7
cd lerobot_xarm7

uv venv --python 3.10
uv sync --extra gello
```

基础依赖包含 `lerobot==0.4.3`（带 Intel RealSense 支持）、`xarm-python-sdk`、`numpy`、`pyyaml` 和 `opencv-python`。`gello` 可选依赖会额外安装 GELLO 软件和 Dynamixel SDK。

### 串口权限

GELLO 示教臂通过串口连接，需要将当前用户加入 `dialout` 组（重新登录后生效）：

```bash
sudo usermod -aG dialout $USER
```

查看 GELLO 串口路径（用于配置文件中的 `teleop.port`）：

```bash
ls /dev/serial/by-id/
```

## 配置

项目在 `config/` 下提供了预置配置文件：

| 采集方式 | 配置文件 |
|---|---|
| GELLO · xArm7 | `config/gello/xarm7_gello_record_config.yaml` |
| 手动拖拽 · xArm7 | `config/manual_mode/xarm7_manual_record_config.yaml` |

### GELLO 配置说明

- `robot.robot_ip` — xArm 控制器 IP（如 `192.168.1.245`）
- `robot.robot_dof` — `7`
- `robot.gripper_type` — `1` 表示 xArm 夹爪
- `teleop.port` — GELLO 串口路径（`/dev/serial/by-id/...`）
- `teleop.joint_ids` / `teleop.joint_signs` — 各型号机械臂的舵机映射与方向
- `teleop.start_joints` — GELLO 校准参考值（角度），应与 xArm SDK 初始点一致
- `teleop.gripper_id` — GELLO 夹爪舵机 ID（`8`；`-1` 表示无夹爪）
- `teleop.realtime_control_fps` — GELLO 到 xArm 的独立实时控制频率，与 `dataset.fps` 分开
- `dataset.root` / `dataset.repo_id` — 数据集保存位置
- `dataset.single_task` — 随每一帧保存的任务描述
- `dataset.fps` / `episode_time_s` / `reset_time_s` — 录制时序参数

> xArm7 的配置已包含正确的关节映射，一般只需要修改串口、IP 和数据集路径。

### 手动拖拽配置说明

- `robot.manual_mode: true` — 开启 xArm 示教模式（关节自由拖动）
- `robot.teach_sensitivity` — 示教灵敏度，有效范围 1–5
- `robot.manual_gripper_speed` — 夹爪速度（每秒归一化位置变化，默认 `0.5`）
- `robot.observe_joint_vel` — 是否在观测中记录关节速度（默认 `false`）
- `robot.enable_logs` — 是否启用每帧耗时和诊断日志（默认 `false`）
- `robot.cameras.camera` — Intel RealSense 相机配置（`serial_number_or_name`、分辨率、fps）
- `dataset.root` / `dataset.repo_id` / `single_task` / `fps` / `episode_time_s` / `reset_time_s` / `num_episodes` — 数据集配置

### 相机配置说明

需要给机器人配置添加相机时，参考 `config/manual_mode/xarm7_manual_record_config.yaml` 中的模板：

```yaml
robot:
  cameras:
    camera:
      type: intelrealsense        # RealSense 类型，不是 opencv
      serial_number_or_name: "148522072685"
      width: 640
      height: 480
      fps: 30
```

- `type` 必须是 `intelrealsense`（RealSense 类型），**不能**写成 `opencv`（普通 USB 相机类型）。
- `serial_number_or_name` 需要先获取 RealSense 相机序列号再填写，否则连接/录制会报错。获取序列号：

```bash
uv run uf-camera-view -l -T realsense     # 列出每台相机的序列号
```

也可以使用 librealsense 自带的 `rs-enumerate-devices`。

## 使用


`Space` 复位并开始，`←` 复位，`Esc` 退出。

#### Guard 延迟实验

以下命令在启用 `min_tcp_z_mm` 安全检测的情况下运行 60 秒，并记录控制周期、
GELLO 读取、安全检测、ServoJ 和完整 `send_action` 耗时：

```bash
uv run uf-robot-teleop \
  --config_path config/gello/xarm7_gello_record_config.yaml \
  --robot.enable_logs=true \
  --fps 60 \
  --guard_latency_experiment=true \
  --experiment_duration_s 60
```

按 `Space` 复位并开始。实验期间可在确保安全的前提下分别经过远离高度下限和接近
高度下限的区域。使用 `tcp_z_guard_backend: local_projection` 时，CSV 的
`guard_path` 会标记 `local_safe`、`local_projected`、`local_hold` 或
`model_fault`，终端也会按路径输出分组统计。结果写入
`logs/gello_guard_latency_<时间>.csv`。

实验结果与分析见 [GELLO 安全高度 Guard 延迟实验记录](docs/gello_guard_latency_experiment_20260817.md)。

完整的抖动修复、数据同步和夹爪 Error 19 排障过程见
[xArm7 + GELLO 平滑安全录制实践](docs/gello_xarm7_smooth_safe_recording_zh.md)。

#### 设置 GELLO TCP 最低高度

先停止其他控制程序，将机械臂 TCP 移到最低安全位置，然后只读当前高度：

```bash
uv run uf-read-tcp-z \
  --config-path config/gello/xarm7_gello_record_config.yaml \
  --margin-mm 5
```

该命令不会移动机械臂。将硬下限填入 `min_tcp_z_mm`；CPU 本地投影会在其上
额外叠加 `tcp_z_soft_margin_mm`。xArm7 GELLO 关节路径会保留全部七个关节目标，
只投影会穿过 TCP 软高度面的运动分量；控制器 Safety Boundary 则在硬下限处
作为最后一道停止保护。

> 该限制只保护 TCP 不低于一个水平面，不能检测机械臂连杆、肘部或夹爪外形与桌子的碰撞，也不能替代急停。更换工具、TCP 偏置、底座或桌面位置后必须重新测量。

### 2. GELLO 数据采集

```bash
# 录制新数据集
uv run uf-lerobot-record --config_path config/gello/xarm7_gello_record_config.yaml

# 在已有数据集上续录
uv run uf-lerobot-record --config_path config/gello/xarm7_gello_record_config.yaml -r

# 可选：后台异步保存 episode
uv run uf-lerobot-record --config_path config/gello/xarm7_gello_record_config.yaml -a
```

按键控制：`Space` 开始当前 episode，`→` 保存，`←` 放弃并重录，`Esc` 停止录制。每个 episode 之间机械臂会自动复位到初始点。

> 采集过程中**机械臂与相机（D435 / D435i）的相对位置必须保持不变**，推理时的相机位置必须与采集时一致。若机械臂或相机发生变化，此前采集的数据将失效。

> 如果数据集目录已存在且未加 `-r`，脚本会询问是覆盖、续录还是取消。

### 3. 手动拖拽数据采集

```bash
./start_manual_record.sh
./start_manual_record.sh -r   # 强制续录；数据集目录不存在时会报错
```

录制脚本（`uf-lerobot-record`）会从配置读取 `dataset.root` 并在录制前检查路径，然后：

- 目录不存在：直接录制新数据集。
- 目录已存在（有效 LeRobot 数据集）且未加 `-r`：交互询问：
  - `o` 覆盖：删除已有数据集，重新录制
  - `r` 续录：保留已有 episode，继续录制
  - `c` 取消
- 目录存在但不完整（缺少必要元数据或数据 parquet 文件）：询问覆盖或取消；非交互运行时直接报错。

加 `-r` 可跳过询问直接续录。注意 `./start_manual_record.sh` 保持原有启动脚本行为——检测到有效数据集会自动续录（相当于 `-r`），如果想看到覆盖/续录的询问，请直接用 `uv run uf-lerobot-record --config_path config/manual_mode/xarm7_manual_record_config.yaml` 运行。

录制时机械臂处于示教模式，实际关节状态会同时作为 observation 和 action 写入数据集。按住 `C` 缓慢闭合夹爪，按住 `O` 缓慢张开。按键控制：`Space` 开始，`→` 保存，`←` 放弃并重录，`Esc` 停止。episode 之间手动复位机械臂。

> **重要：每个 episode 开始时，必须等待机械臂复位完成后再等待 5 秒，然后才能开始操作；或者确认控制台打印 `Start Recording` 后再开始操作。** 这样可以避免机械臂复位控制指令与操作指令冲突导致报错。

### 4. 策略训练

```bash
uv run lerobot-train --policy act --dataset ufactory/xarm7_gello_datas
```

带完整训练参数的示例（每 `save_freq` 步保存一次 checkpoint 到 `output_dir`）：

```bash
uv run lerobot-train \
  --dataset.root=/home/<user>/lerobot_datas/record/ufactory/xarm7_gello_datas \
  --dataset.repo_id=ufactory/xarm7_gello_datas \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id=ufactory/xarm7_gello_datas \
  --output_dir=/home/<user>/lerobot_datas/train/xarm7_gello_datas \
  --job_name=xarm7_gello_datas \
  --steps=800000 \
  --batch_size=8 \
  --save_freq=20000
```

### 5. 策略推理

```bash
uv run uf-lerobot-eval \
  --config_path config/gello/xarm7_gello_record_config.yaml \
  --policy.path /path/to/train/output/checkpoints/last/pretrained_model/
```

`←` / `→` 复位，`Esc` 停止。

### 6. 回放已录制 episode

将手动拖拽数据集的绝对关节状态（`observation.state`）回放到 xArm7。脚本按数据集 FPS（默认 30）将状态作为**绝对目标值**发送，不做差分或累加，因此运动轨迹与录制时一致：

```bash
# 默认回放第一条
uv run uf-lerobot-replay \
  --dataset-root /path/to/xarm7_manual_datas \
  --robot-ip 192.168.1.245

# 跳过交互确认（无人值守）
uv run uf-lerobot-replay --dataset-root /path/to/xarm7_manual_datas --robot-ip 192.168.1.245 --yes

# 回放其他 episode
uv run uf-lerobot-replay --dataset-root /path/to/xarm7_manual_datas --robot-ip 192.168.1.245 --episode-index 3
```

回放开始前机械臂会先移动到 xArm SDK 初始点，播放结束后保持最后一帧姿态并断开连接。执行前请确认工作空间无障碍物，且数据中的初始姿态与当前设备一致。

## 重要：机械臂开关机事项

### 开机

1. **使用网线将电脑连接到机械臂控制器。**
2. **参考控制器上标注的机械臂 IP，将电脑以太网接口配置到同一网段**（例如 `192.168.1.xxx`）。
3. **在浏览器中访问 `http://192.168.1.245:18333/`**，应出现控制台界面。
4. **操作机械臂前，抬起急停按钮。**

### 关机

1. **先通过网页控制将机械臂返回到初始位置。**
2. **再按下急停按钮。**
3. **关闭控制器电源。**

## 工具

### 摄像头查看器

```bash
uv run uf-camera-view -l                # 列出所有摄像头
uv run uf-camera-view -T realsense      # 查看 RealSense 摄像头
```

网页预览会扫描所有 RealSense 摄像头，默认使用 `640x480`、`30fps`，页面中可勾选设备切换或同时显示多路画面：

```bash
uv run uf-realsense-view
```

脚本使用 LeRobot 的 `RealSenseCamera`。如果当前环境中的 OpenCV 是 LeRobot 默认的
headless 版本，会自动启动网页预览。同一台机器上打开
`http://127.0.0.1:8765/`；从其他机器访问时，请将 `127.0.0.1` 替换为运行脚本机器的实际 IP。
网页服务默认监听 `0.0.0.0`，也可以用 `--host 127.0.0.1` 限制为本机访问；还可以通过
`--backend opencv` 强制使用 OpenCV 窗口。

也可以只打开指定设备；需要多个设备时重复 `--serial`：

```bash
uv run uf-realsense-view \
  --serial 148522072685 --width 640 --height 480 --fps 30

uv run uf-realsense-view \
  --serial 148522072685 --serial SECOND_CAMERA_SERIAL
```

OpenCV 窗口按 `q` 或 `Esc` 退出；网页模式按 `Ctrl+C` 退出。无桌面环境时可以用
`--no-display` 检查是否能持续取帧。

### LeRobot 数据集工具

```bash
# 查看索引为 17 的 episode
uv run lerobot-dataset-viz \
  --root=/path/to/record/ufactory/xarm7_manual_datas \
  --repo-id ufactory/xarm7_manual_datas \
  --display-compressed-images true \
  --episode-index 17

# 删除索引为 18 和 19 的 episode
uv run lerobot-edit-dataset \
  --root=/path/to/record/ufactory/xarm7_manual_datas \
  --repo_id ufactory/xarm7_manual_datas \
  --new_repo_id ../xarm7_manual_datas_new \
  --operation.type delete_episodes \
  --operation.episode_indices "[18, 19]"

# 合并数据集
uv run lerobot-edit-dataset \
  --root=/path/to/record \
  --repo_id ufactory/xarm7_datas_merge \
  --operation.type merge \
  --operation.repo_ids "['ufactory/xarm7_datas_1', 'ufactory/xarm7_datas_2']"
```

## 项目结构

```
lerobot_xarm7/
├── config/
│   ├── gello/                     # xArm7 GELLO 录制配置
│   └── manual_mode/               # xArm7 手动拖拽录制配置
├── src/lerobot_robot_ufactory/
│   ├── robots/
│   │   └── uf_robot/              # xArm 控制（关节/笛卡尔空间、示教模式）
│   ├── teleoperators/
│   │   ├── base_teleop/           # 遥操作基类
│   │   └── gello_teleop/          # GELLO（Dynamixel 示教臂）
│   ├── scripts/
│   │   ├── uf_robot_teleop.py     # 遥操作测试
│   │   ├── uf_lerobot_record.py   # 数据采集（含手动模式）
│   │   ├── uf_lerobot_eval.py     # 策略推理
│   │   ├── uf_lerobot_replay.py   # episode 回放
│   │   └── uf_camera_view.py      # 摄像头查看器
│   └── configs/parser.py          # 配置加载 / CLI 覆盖
├── start_manual_record.sh         # 手动拖拽启动脚本
├── pyproject.toml
├── README.md
└── README_ZH.md
```

## 重要提示

- 提供的配置都是**示例**：请根据实际硬件修改 IP、串口、相机序列号、数据集路径和任务描述。
- GELLO 数据采集与推理时，机械臂与相机的相对位姿必须保持一致。
- LeRobot 中扩散策略（Diffusion Policy）的默认参数主要面向仿真，**未针对真实机器人优化**，需要根据任务自行调整。
- 回放或推理前，请确认工作空间无障碍物，并保证机械臂初始姿态与录制数据一致。

## 许可证

本项目基于 Apache License 2.0 发布，详见 [LICENSE](LICENSE) 文件。
