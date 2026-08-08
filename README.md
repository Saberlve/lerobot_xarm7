# UFACTORY xArm7 · LeRobot (GELLO / Manual Drag)

> [中文版本](README_ZH.md)

UFACTORY xArm integration with the [LeRobot](https://github.com/huggingface/lerobot) framework, focused on two data-collection workflows:

- **GELLO** — joint-space teleoperation with a Dynamixel leader arm
- **Manual drag** — record demonstrations by freely moving the arm in xArm teach mode

Collected data is stored in the standard LeRobot dataset format and can be used for imitation learning (ACT / Diffusion Policy, etc.) and real-time policy inference.

## Features

- 🤖 UFACTORY xArm7 control
- 🎮 GELLO joint-space teleoperation (Dynamixel leader arm)
- ✋ Manual drag recording via xArm teach mode
- 📷 Intel RealSense camera observation (D435 / D435i)
- 📊 LeRobot-compatible dataset recording & management
- 🧠 Imitation learning training and policy inference
- ▶️ Episode replay for recorded manual demonstrations

## Requirements

- Ubuntu 22.04 / 24.04
- Python >= 3.10
- CUDA >= 12.0 (recommended for GPU training)
- UFACTORY xArm7 and its controller
- GELLO arm (FTDI USB serial)
- Intel RealSense D435 / D435i (for camera observations)

## Installation

```bash
git clone https://git.weiyantech.cn/wangshuxun/Xarm-DataCollection.git lerobot_xarm7
cd lerobot_xarm7

uv venv --python 3.10
uv sync --extra gello
```

The base dependencies include `lerobot==0.4.3` (with Intel RealSense support), `xarm-python-sdk`, `numpy`, `pyyaml`, and `opencv-python`. The `gello` extra adds the GELLO software and Dynamixel SDK.

### Serial port permission

The GELLO arm connects over a serial port, so add your user to the `dialout` group (re-login afterwards):

```bash
sudo usermod -aG dialout $USER
```

Find the GELLO serial port path (used as `teleop.port` in the configs):

```bash
ls /dev/serial/by-id/
```

## Configuration

Predefined configs are provided under `config/`:

| Workflow | Config |
|---|---|
| GELLO · xArm7 | `config/gello/xarm7_gello_record_config.yaml` |
| Manual drag · xArm7 | `config/manual_mode/xarm7_manual_record_config.yaml` |

### GELLO config

- `robot.robot_ip` — xArm controller IP (e.g. `192.168.1.245`)
- `robot.robot_dof` — `7`
- `robot.gripper_type` — `1` for the xArm gripper
- `teleop.port` — GELLO serial port (`/dev/serial/by-id/...`)
- `teleop.joint_ids` / `teleop.joint_signs` — per-arm servo mapping and direction
- `teleop.start_joints` — GELLO calibration reference, should match the xArm SDK initial point (degrees)
- `teleop.gripper_id` — GELLO gripper servo ID (`8`; `-1` disables it)
- `dataset.root` / `dataset.repo_id` — where the dataset is stored
- `dataset.single_task` — task description saved with each frame
- `dataset.fps` / `episode_time_s` / `reset_time_s` — recording timing

> The xArm7 config already contains the correct joint mapping; only edit the port, IP, and dataset fields for your setup.

### Manual-drag config

- `robot.manual_mode: true` — enable xArm teach mode (joint free-drive)
- `robot.teach_sensitivity` — teaching sensitivity, valid range 1–5
- `robot.manual_gripper_speed` — gripper velocity in normalized position per second (default `0.5`)
- `robot.observe_joint_vel` — record joint velocities in observations (`false` by default)
- `robot.cameras.camera` — Intel RealSense camera (`serial_number_or_name`, resolution, fps)
- `dataset.root` / `dataset.repo_id` / `single_task` / `fps` / `episode_time_s` / `reset_time_s` / `num_episodes` — dataset settings

### Camera configuration

When adding a camera to the robot config, use the template in
`config/manual_mode/xarm7_manual_record_config.yaml`:

```yaml
robot:
  cameras:
    camera:
      type: intelrealsense        # RealSense type, NOT opencv
      serial_number_or_name: "148522072685"
      width: 640
      height: 480
      fps: 30
```

- `type` must be `intelrealsense` (RealSense), **not** `opencv` (generic USB camera).
- `serial_number_or_name` must be filled with the actual RealSense serial number **obtained beforehand**, otherwise connecting/recording fails. Get it with:

```bash
uv run uf-camera-view -l -T realsense     # prints each camera's serial number
```

or with the librealsense tool `rs-enumerate-devices`.

## Usage

### 1. GELLO teleop test

Test the GELLO → robot control loop without recording:

```bash
uv run uf-robot-teleop --config_path config/gello/xarm7_gello_record_config.yaml
uv run uf-robot-teleop --config_path config/gello/xarm7_gello_record_config.yaml --fps 60  # optional loop rate
```

`Space` reset & start, `←` reset, `Esc` exit.

### 2. GELLO data collection

```bash
# Record a new dataset
uv run uf-lerobot-record --config_path config/gello/xarm7_gello_record_config.yaml

# Continue recording on an existing dataset
uv run uf-lerobot-record --config_path config/gello/xarm7_gello_record_config.yaml -r

# Optional: save episodes in the background
uv run uf-lerobot-record --config_path config/gello/xarm7_gello_record_config.yaml -a
```

Controls: `Space` start the episode, `→` save it, `←` discard and re-record it, `Esc` stop recording. The arm resets to its initial point between episodes.

> During collection the **relative position between the robot arm and the camera must not change**, and the camera setup at inference time must match the one used during collection. If the arm or camera moves, previously collected data becomes invalid.

> If the dataset root already exists and `-r` is not given, the script asks whether to overwrite it, resume, or cancel.

### 3. Manual drag recording

```bash
./start_manual_record.sh
./start_manual_record.sh -r   # force resume; fails if the dataset directory does not exist
```

The record script (`uf-lerobot-record`) reads `dataset.root` from the config and checks the path before recording, then:

- Directory does not exist → records a new dataset.
- Directory already exists (valid LeRobot dataset) and no `-r` was given → asks interactively:
  - `o` overwrite: delete the existing dataset and record a new one
  - `r` resume: keep existing episodes and continue recording
  - `c` cancel
- Directory exists but is incomplete (missing required metadata or data parquet files) → asks to overwrite it or cancel; non-interactive runs error out instead.

Pass `-r` to resume directly without asking. Note that `./start_manual_record.sh` keeps its original launcher behavior — it automatically resumes an existing valid dataset (equivalent to `-r`), so run `uv run uf-lerobot-record --config_path config/manual_mode/xarm7_manual_record_config.yaml` directly if you want to see the overwrite/resume prompt.

During recording the arm is in teach mode: the actual joint state is written as both the observation and the action. Hold `C` to slowly close the gripper and `O` to slowly open it. Controls: `Space` start, `→` save, `←` discard & re-record, `Esc` stop. Reset the arm manually between episodes.

> **IMPORTANT: At the start of every episode, wait until the arm has finished resetting and then wait another 5 seconds before operating it, or wait until the console prints `Start Recording` before operating it.** This prevents reset commands from conflicting with operation commands and causing errors.

### 4. Policy training

```bash
uv run lerobot-train --policy act --dataset ufactory/xarm7_gello_datas
```

Example with explicit training parameters (checkpoints are saved every `save_freq` steps into `output_dir`):

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

### 5. Policy inference

```bash
uv run uf-lerobot-eval \
  --config_path config/gello/xarm7_gello_record_config.yaml \
  --policy.path /path/to/train/output/checkpoints/last/pretrained_model/
```

`←` / `→` reset, `Esc` stop.

### 6. Replay recorded episodes

Replay the absolute joint states (`observation.state`) of a manual-drag episode on an xArm7. States are sent as absolute targets at the dataset FPS (default 30), so the motion matches the recording:

```bash
uv run uf-lerobot-replay \
  --dataset-root /path/to/xarm7_manual_datas \
  --robot-ip 192.168.1.245

# Skip the interactive confirmation (non-interactive use)
uv run uf-lerobot-replay --dataset-root /path/to/xarm7_manual_datas --robot-ip 192.168.1.245 --yes

# Replay another episode
uv run uf-lerobot-replay --dataset-root /path/to/xarm7_manual_datas --robot-ip 192.168.1.245 --episode-index 3
```

The robot first moves to the xArm SDK initial point, then replays the episode and stays at the last state. Make sure the workspace is clear and the recorded initial pose matches the current arm setup.

## IMPORTANT: Robot Power-On and Power-Off

### Power-on

1. **Connect the computer to the robot controller with an Ethernet cable.**
2. **Configure the computer's Ethernet interface to the same IP subnet as the robot IP shown on the controller** (for example, `192.168.1.xxx`).
3. **Open `http://192.168.1.245:18333/` in a browser.** The controller console should be displayed.
4. **Release the emergency-stop button before operating the robot.**

### Power-off

1. **Use the web console to return the arm to its initial position.**
2. **Press the emergency-stop button.**
3. **Turn off the controller power.**

## Tools

### Camera viewer

```bash
uv run uf-camera-view -l                # list cameras
uv run uf-camera-view -T realsense      # view RealSense cameras
```

### LeRobot dataset tools

```bash
# View episode 17
uv run lerobot-dataset-viz \
  --root=/path/to/record/ufactory/xarm7_manual_datas \
  --repo-id ufactory/xarm7_manual_datas \
  --display-compressed-images true \
  --episode-index 17

# Delete episodes 18 and 19
uv run lerobot-edit-dataset \
  --root=/path/to/record/ufactory/xarm7_manual_datas \
  --repo_id ufactory/xarm7_manual_datas \
  --new_repo_id ../xarm7_manual_datas_new \
  --operation.type delete_episodes \
  --operation.episode_indices "[18, 19]"

# Merge datasets
uv run lerobot-edit-dataset \
  --root=/path/to/record \
  --repo_id ufactory/xarm7_datas_merge \
  --operation.type merge \
  --operation.repo_ids "['ufactory/xarm7_datas_1', 'ufactory/xarm7_datas_2']"
```

## Project structure

```
lerobot_xarm7/
├── config/
│   ├── gello/                     # xArm7 GELLO record config
│   └── manual_mode/               # xArm7 manual-drag record config
├── src/lerobot_robot_ufactory/
│   ├── robots/
│   │   └── uf_robot/              # xArm control (joint/cartesian, teach mode)
│   ├── teleoperators/
│   │   ├── base_teleop/           # shared teleop base class
│   │   └── gello_teleop/          # GELLO (Dynamixel leader arm)
│   ├── scripts/
│   │   ├── uf_robot_teleop.py     # teleop test loop
│   │   ├── uf_lerobot_record.py   # data collection (incl. manual mode)
│   │   ├── uf_lerobot_eval.py     # policy inference
│   │   ├── uf_lerobot_replay.py   # episode replay
│   │   └── uf_camera_view.py      # camera viewer
│   └── configs/parser.py          # config loading / CLI overrides
├── start_manual_record.sh         # manual-drag launcher
├── pyproject.toml
├── README.md
└── README_ZH.md
```

## Important notes

- The provided configs are **examples**: edit IPs, serial ports, camera serials, dataset paths, and task descriptions to match your hardware.
- For GELLO data, keep the robot–camera relative pose identical between collection and inference.
- The LeRobot default parameters for diffusion policies are mostly designed for simulation and are **not optimized for real robots** — tune them for your task.
- Check the workspace for obstacles and keep the arm's initial pose consistent with the recorded data before replay or inference.

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
