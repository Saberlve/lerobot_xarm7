# UFACTORY xArm7 · LeRobot（GELLO / 手动拖拽）

## GELLO 恒流与阻尼

main 仅运行恒流与定幅阻尼；模型重力补偿实现保存在 `feature/gello-gravity-compensation` 分支。
当前配置 J2 为 -40 mA，J4 为 +80 mA；J3/J7 为与运动方向相反的 2 mA 阻尼，速度死区为 0.05 rad/s；其余轴零电流。实际运行值以配置文件为准。
参数在 `config/current_control/gello_A_working.yaml`。启动保留 2 秒缓升、电流限速、温度保护和通信看门狗。
网页保留模型查看与电流控制，默认离线，不会自动启用电机：

```bash
.venv/bin/python -m lerobot_robot_ufactory.current_control.web.tuning_web --profile config/current_control/gello_A_working.yaml --port 8765
```

遥操作使用 `config/gello/xarm7_gello_teleop_current.yaml`。
完整 chips 录制使用 `config/gello/xarm7_gello_record_xense_potato_chips.yaml`，控制 30 Hz、数据集 15 FPS，含相机及双 Photon。
配置项为 `teleop.current_control`，不再支持重力增益。网页失联超过 3 秒自动卸力。
网页可逐轴输入恒流（正负表示电机出力方向）与阻尼（非负值），点击“应用并同步配置”后生效并自动写入当前设备配置；运行时按当前电流变化率过渡。运行电流变化率修改后也自动保存为 `running_current_slew_a_s`。网页重启与下一次遥操作都会读取保存值，使用同一设备配置的默认遥操作和录制配置不再重复设置变化率。同步失败时网页明确提示，可点击“重新同步参数到配置”重试。调参完成后先立即卸力，再启动遥操作，避免串口冲突。
Photon 离线 Mesh3DFlow 按传感器复用 solver，整批 episode 完成或异常时释放。

UFACTORY xArm 与 [LeRobot](https://github.com/huggingface/lerobot) 框架的集成项目，专注于两种数据采集方式：

- **GELLO** — 使用 Dynamixel 示教臂的关节空间遥操作
- **手动拖拽** — 在 xArm 示教模式下自由拖动机械臂录制演示

采集的数据以标准 LeRobot 数据集格式保存，可用于模仿学习训练（ACT / Diffusion Policy 等）和实时策略推理。

## 功能特性

- 🤖 UFACTORY xArm7 控制
- 🎮 GELLO 关节空间遥操作（Dynamixel 示教臂）
- ✋ xArm 示教模式手动拖拽采集
- 📷 Intel RealSense 相机观测（D435 / D435i）
- 双 Xense Photon 触觉图像采集：[示例配置](config/gello/xarm7_gello_record_xense_photon_config.yaml)
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
- `robot.record_space` — 数据集保存格式：`"joint"`（默认）记录 J1..J7 关节角（弧度）；`"tcp"` 记录经 FK 转换的 TCP 位姿（`pose.x/y/z` 单位 mm，旋转用连续 6D 表示 `pose.r11/r21/r31/r12/r22/r32`，即旋转矩阵的前两列，参考 [Zhou et al., CVPR 2019](https://arxiv.org/abs/1812.07035)，不存在欧拉角/轴角的跳变问题）；`"both"` 同时记录关节角和 TCP 位姿。GELLO 控制仍保持在关节空间
- `robot.gripper_type` — `2` 表示 xArm Gripper G2
- `robot.gripper_speed` — G2 开合速度，单位 mm/s（范围 `15`–`225`，当前配置为 `100`）
- `robot.gripper_force` — G2 夹持力（范围 `1`–`100`，当前配置为 `50`）
- `teleop.port` — GELLO 串口路径（`/dev/serial/by-id/...`）
- `teleop.joint_ids` / `teleop.joint_signs` — 各型号机械臂的舵机映射与方向
- `teleop.start_joints` — GELLO 校准参考值（角度），应与 xArm SDK 初始点一致
- `teleop.gripper_id` — GELLO 夹爪舵机 ID（`8`；`-1` 表示无夹爪）
- `teleop.gripper_open_deg` / `teleop.gripper_close_deg` — GELLO 舵机的开闭标定角，与 G2 的 0–84 mm 行程相互独立
- `teleop.realtime_control_fps` — GELLO 到 xArm 的独立实时控制频率，与 `dataset.fps` 分开
- `dataset.root` / `dataset.repo_id` — 数据集保存位置
- `dataset.single_task` — 随每一帧保存的任务描述
- `dataset.fps` — 动作、状态及其他数值字段的行采样频率；完整相机流使用各自 `robot.cameras.<name>.fps`
- `episode_time_s` / `reset_time_s` — episode 与复位时长
- 相机独立帧率、触觉视频与无损 Mesh3DFlow：触觉按相机帧率保存 H.264/YUV420P 视频；编码前计算并独立无损保存完整 Mesh3DFlow，已有数据集保持原样

凡是保存完整触觉流，每行都新增 `observation.<camera>.tactile_range`，类型为
`int64[2]`，以 `[start_index, end_index)` 关联该 episode 的 `samples.parquet`。
不开启 Mesh 时，范围关联触觉图像；开启 Mesh 时，同一范围同时关联图像和
Mesh3DFlow。同频和不同频均使用这一规则。没有新帧时保存 `[k, k)`，该行观测的
代表图像通过 `timestamps/episode_<index>.parquet` 中的
`representative_tactile_index` 另行关联。
如果首条动作使用了录制开始前的图像，该原始样本会先保存到完整相机流，
但不计入首个动作窗口。例如起始代表帧索引为 0，首窗口没有新帧时范围为
`[1, 1)`；代表帧也会参与启用的 Mesh 计算和视频编码。

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
历史 GELLO 读取、安全检测和完整 `send_action` 耗时诊断：

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
uv run record --config_path config/gello/xarm7_gello_record_config.yaml

# 在已有数据集上续录
uv run record --config_path config/gello/xarm7_gello_record_config.yaml -r

# 可选：后台异步保存 episode
uv run record --config_path config/gello/xarm7_gello_record_config.yaml -a
```

按键控制：`Space` 开始当前 episode，`→` 保存，`←` 放弃并重录，`Esc` 停止录制。每个 episode 之间机械臂会自动复位到初始点。

每条 episode 录制结束后，等待图像写入完成，计算启用的离线 Mesh3DFlow，
编码该条视频，并保存 LeRobot 数据、触觉原始流及时间对应关系。默认等待当前
episode 保存完成后再开始下一条；使用 `-a` 时，每条 episode 在后台完成推理、
编码和保存，退出前等待所有保存任务完成。

同步超时只丢弃当前 episode，等待 `Space`（无键盘监听时按 `Enter`）重录。
后续录制异常不会影响此前已经保存的 episode。

旧版本录制留下的未处理原始 checkpoint，需要使用原配置运行以下命令后再续录，
此命令不连接录制设备：

```bash
uv run record --config_path config/gello/xarm7_gello_base.yaml --postprocess-only
```

Photon 的离线 Mesh3DFlow 使用 NVIDIA GPU，两个传感器各自按帧顺序并行计算。
安装后处理依赖时启用 `xense-gpu`，并保留需要的其他 extras，例如：

```bash
uv sync --extra gello --extra xense-gpu
```

不要同时安装 `onnxruntime` 与 `onnxruntime-gpu`，两者共享同一个 Python 包。
GPU 版会复用 PyTorch 的 CUDA/cuDNN 动态库，创建 solver 后检查实际后端，
并在控制台显示 `GPU inference (CUDAExecutionProvider)`。GPU 不可用时后处理
报错，原始数据保留供修复后重试。首次 GPU 推理可能需要较长的初始化时间；
GPU 与此前 CPU 结果可能存在微小浮点差异，保存仍保留 SDK 输出的原始精度。
CUDA/cuDNN 兼容要求参见 [ONNX Runtime 官方说明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

> 采集过程中**机械臂与相机（D435 / D435i）的相对位置必须保持不变**，推理时的相机位置必须与采集时一致。若机械臂或相机发生变化，此前采集的数据将失效。

默认 `synchronize: true`。每个已保存的 GELLO episode 会在数据集根目录的
`timestamps/` 下额外写入 state、GELLO action 和相机读取到达时间的 Parquet
sidecar，并在控制台输出同步统计；它不改变 LeRobot 训练数据的 schema。若只需
普通录制，可在 YAML 顶层设置 `synchronize: false` 关闭这些文件和统计。

所有普通 RGB 相机（包括 RealSense）都按新鲜 `async_read` 帧到达主机的时间，
选择不晚于动作发送起点的最新帧。RealSense 使用 LeRobot 原有相机后端。
单帧选择不等待下一帧；原始相机时间窗口仍可等待窗口结束的标记帧。
主机接收时间保存在动作时间 sidecar，以及原始 RGB 流 `samples.parquet` 的
`camera_timing_json` 中。

> 如果数据集目录已存在且未加 `-r`，脚本会询问是覆盖、续录还是取消。

### 3. 手动拖拽数据采集

```bash
./start_manual_record.sh
./start_manual_record.sh -r   # 强制续录；数据集目录不存在时会报错
```

`record` 命令会从配置读取 `dataset.root` 并在录制前检查路径，然后：

- 目录不存在：直接录制新数据集。
- 目录已存在（有效 LeRobot 数据集）且未加 `-r`：交互询问：
  - `o` 覆盖：删除已有数据集，重新录制
  - `r` 续录：保留已有 episode，继续录制
  - `c` 取消
- 目录存在但不完整（缺少必要元数据或数据 parquet 文件）：询问覆盖或取消；非交互运行时直接报错。

加 `-r` 可跳过询问直接续录。注意 `./start_manual_record.sh` 保持原有启动脚本行为——检测到有效数据集会自动续录（相当于 `-r`），如果想看到覆盖/续录的询问，请直接用 `uv run record --config_path config/manual_mode/xarm7_manual_record_config.yaml` 运行。

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
uv run replay \
  --dataset-root /path/to/xarm7_manual_datas \
  --robot-ip 192.168.1.245

# 跳过交互确认（无人值守）
uv run replay --dataset-root /path/to/xarm7_manual_datas --robot-ip 192.168.1.245 --yes

# 回放其他 episode
uv run replay --dataset-root /path/to/xarm7_manual_datas --robot-ip 192.168.1.245 --episode-index 3
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
│   ├── datasets/                 # 数据集存储、原生相机流和离线处理
│   │   ├── native_dataset.py     # 数据集读写与时间映射
│   │   ├── camera_streams.py     # 相机流方案、采样与编码
│   │   ├── tactile_indices.py   # 触觉范围索引
│   │   ├── stream_recorder.py   # 后台写入与存储事务
│   │   ├── deferred_mesh.py     # episode 离线 Mesh 计算流程
│   │   ├── episode_images.py   # 临时图像校验与清理
│   │   └── raw_episodes.py      # 原始 episode 保存、恢复与后处理
│   ├── cameras/                 # RGB 驱动、时间戳与采样队列
│   ├── tactile/                 # 触觉接口、Photon 驱动与单帧 SDK 推理
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


## GELLO 网页录制工作台

使用 `.venv/bin/uf-lerobot-record-web --port 8769` 启动统一配置管理、录制控制和相机预览。支持浏览器键盘与仅 J7 模式，Photon 预览默认关闭。参见 [网页录制说明](docs/recording_web.md)。

选择并保存配置后，“数据后处理”面板会自动检测该数据目录中的待处理原始条，每 5 秒刷新，也可手动刷新。点击“单独后处理”可在不连接机器人、GELLO 或相机的情况下处理这些数据，并显示已转换条数、当前阶段和各相机的帧数进度。后处理与录制互斥，网页断线不会终止处理；失败后原始数据保留，可修复问题后重试。`defer_processing` 会话退出时的自动后处理也显示同样的进度。
