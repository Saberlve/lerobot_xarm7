# 双 Xense Photon 触觉图像采集

接入代码：`src/lerobot_robot_ufactory/cameras/xense_photon_camera/`。
配置类型：`photon`，每个传感器对应一个相机配置。
可将下面的两路配置复制到其他遥操作或手动采集 YAML 的 `robot.cameras` 中。

## 1. 安装 SDK

在本仓库根目录执行以下命令，将 SDK 及其运行依赖追加到已有的 `.venv`：

```bash
uv pip install --reinstall-package xensesdk 'xensesdk>=2.0.0,<3' \
  --default-index https://mirrors.aliyun.com/pypi/simple/
```

不要只为安装 Photon SDK 执行 `uv sync --extra xense`：`uv sync` 会将虚拟环境收敛为
当前选中的依赖集，可能移除已有的 GELLO、开发工具或其他可选依赖。

如果使用 pip 管理环境，在已有 LeRobot 环境中执行：

```bash
pip install -e '.[xense]'
```

SDK 是可选依赖，不使用 Photon 时无需安装。依赖限定为 Xense SDK 2.x；
使用 SDK 提供的 Python/平台对应二进制包，建议沿用本仓库的 Python 3.10 环境。

默认 `disable_infer: true`、`use_gpu: false`，只采集图像。若要保存 marker 三维位移，
必须设置 `disable_infer: false` 和 `save_marker_motion_3d: true`；SDK 会在同一次读取中返回
RGB、`Marker3DFlow` 和 `TimeStamp`。先使用 CPU 验证帧率；需要时再按 Xense SDK 文档安装
ONNX/GPU 推理依赖并将 `use_gpu` 设为 `true`。
不会自动调用接触标定。如需厂家提供的标定文件，为每个传感器设置 `config_path`（文件或目录）。

参考 [Xense 官方仓库](https://github.com/XenseRobotics/xensesdk)；接口核对版本：
`d73909c23843f4e2429db21b7f0b362280248835`，包括 `Examples/example_local_basic.py`。
该 GitHub 仓库提供文档和示例，实际运行还需要安装 `xensesdk` 包。

## 2. USB 权限和序列号

Ubuntu 如有 USB 权限问题，可安装本仓库的规则：

```bash
sudo groupadd -f xense
sudo usermod -aG xense "$USER"
sudo install -m 644 rules/99-xense.rules /etc/udev/rules.d/99-xense.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

重新登录使用户组生效，然后重新插拔传感器。关闭占用传感器的 Xense Studio 或其他采集程序。
扫描序列号（不连接或移动机械臂）：

```bash
.venv/bin/python -c 'from xensesdk import Sensor; print(Sensor.scanSerialNumber())'
```

分别确认夹爪左侧和右侧的序列号。可逐个接入以确定对应关系；不要用可能随插拔变化的设备编号代替序列号。

## 3. 配置两路 Photon

完整 GELLO 示例：`config/gello/xarm7_gello_record_xense_photon_config.yaml`。
将其中两个 `REPLACE_WITH_..._PHOTON_SN` 替换为实际序列号，并检查机械臂 IP、
GELLO 串口、RealSense 序列号和数据集目录。示例沿用现有 GELLO 配置的这些参数。

```yaml
robot:
  # 其余机器人参数沿用现有配置
  # state、所有 RGB 相机及两路 Photon 的软件同步门限
  sync_max_skew_ms: 20
  sync_pair_max_skew_ms: 20
  sync_wait_ms: 40
  sync_history_size: 90
  cameras:
    xense_left:
      type: photon
      serial_number: "REPLACE_WITH_LEFT_PHOTON_SN"
      width: 400
      height: 700
      fps: 30
      output_type: Rectify
      color_mode: rgb
      disable_infer: false
      use_gpu: false
      save_marker_motion_3d: true
      marker_rows: 35
      marker_cols: 20
    xense_right:
      type: photon
      serial_number: "REPLACE_WITH_RIGHT_PHOTON_SN"
      width: 400
      height: 700
      fps: 30
      output_type: Rectify
      color_mode: rgb
      disable_infer: false
      use_gpu: false
      save_marker_motion_3d: true
      marker_rows: 35
      marker_cols: 20
```

- `width` / `height`：保存图像的尺寸，SDK 图像会缩放到此大小；400×700 是示例，
  不是对 Photon 原生分辨率的承诺。根据实机图像宽高比调整，避免拉伸。
- `fps`：后台读取 SDK 的目标频率，不设置传感器硬件帧率。
- `output_type`：默认 `Rectify` 校正图像；也支持 `Raw`、`Difference` 三通道图像。
- `color_mode`：SDK 返回 BGR，默认转换为 LeRobot 使用的 RGB；采集数据集建议保持 `rgb`。
- `save_marker_motion_3d`：设为 `true` 时实时保存 SDK 的 `Marker3DFlow` 三维 marker 位移，
  并要求 `disable_infer: false`。
- `marker_rows` / `marker_cols`：位移场的固定网格尺寸，默认 35×20。SDK 返回的实际形状必须
  与此一致，否则采集会立刻报错，避免把错误维度的数据写入数据集。
- `sync_max_skew_ms`：state 时间锚点与每一路视觉样本允许的最大主机时间误差。
- `sync_pair_max_skew_ms`：所有保存的视觉样本（RGB 和 Photon）的两两最大主机时间误差。
- `sync_wait_ms`：锚点之后等待新帧的最长时间。任一路在此时限内没有合格帧则本录制周期失败。
- `sync_history_size`：普通 RGB 相机的主机时间戳队列长度。Photon 使用各自相机配置中的同名值。
- `timeout_ms`：Photon 读取等待及断开线程等待上限，默认 2000 毫秒。
- `max_frame_age_ms`：本地图像缓存有效期，默认 1000 毫秒；超过后等待新图像，超时报错。
- `config_path`：可选的 SDK 标定配置路径。

两个配置不能使用同一序列号，否则相机工厂会在连接前报错。

## 4. 单独验证触觉采集

先在 YAML 中填好序列号，然后执行以下检查。它只打开两路触觉传感器，不连接机械臂：

```bash
.venv/bin/python - <<'PY'
import draccus
import yaml
from lerobot.cameras.configs import CameraConfig
from lerobot_robot_ufactory.cameras.utils import make_cameras_from_configs

with open('config/gello/xarm7_gello_record_xense_photon_config.yaml') as f:
    raw = yaml.safe_load(f)
configs = {
    name: draccus.decode(CameraConfig, cfg)
    for name, cfg in raw['robot']['cameras'].items()
    if cfg['type'] == 'photon'
}
cameras = make_cameras_from_configs(configs)
try:
    for camera in cameras.values():
        camera.connect()
    for name, camera in cameras.items():
        frame = camera.async_read()
        print(name, frame.shape, frame.dtype)
finally:
    for camera in cameras.values():
        if camera.is_connected:
            camera.disconnect()
PY
```

确认两路均有图像后启动完整采集（会按现有流程连接机械臂和遥操作器）：

```bash
.venv/bin/record \
  --config_path config/gello/xarm7_gello_record_xense_photon_config.yaml
```

默认数据集的两路字段为 `observation.images.xense_left` 和
`observation.images.xense_right`。完整示例以 25 Hz 保存，每路独立读取最新图像，
并接入现有网页预览。

## 数据范围与同步限制

启用 `save_marker_motion_3d` 后，每路 Photon 保存三项数据：

- `observation.images.xense_left` / `xense_right`：RGB 触觉图像；
- `observation.xense_left.marker_motion_3d` / `xense_right.marker_motion_3d`：`float32`
  的 `(35, 20, 3)` marker 三维位移场；
- `observation.xense_left.sensor_timestamp` / `xense_right.sensor_timestamp`：SDK 时间戳。

每路 Photon 的图像、位移场和 SDK 时间戳由同一次 `selectSensorInfo(Rectify,
Marker3DFlow, TimeStamp)` 调用取得，因此必定对应同一 SDK 帧。两路 Photon 各保留最近
`sync_history_size` 个完整样本。RealSense、Azure 等普通 RGB 后端没有统一的原生帧时间戳接口；
本项目为每路这类相机启动唯一的 `async_read` 消费线程，在新帧到达主机时记录同一单调时钟并保留
等长队列。接入新的 RGB 后端时，`async_read` 必须满足 LeRobot 的“返回新帧”语义；若后端只会
重复返回缓存而没有新帧事件或原生时间戳，不能把它用于严格软件配对。

录制时以 xArm TCP 30000 RT 状态报告的主机接收时刻为共同锚点，所有 RGB 相机和两路 Photon
都只从自己的队列取该锚点附近的帧。`sync_max_skew_ms` 限制 state 到每路视觉样本的误差，
`sync_pair_max_skew_ms` 还限制任意两路视觉样本之间的误差；`sync_wait_ms` 限定等待锚点后新帧
的时长。状态报告本身超过 `sync_max_skew_ms` 未更新也会拒绝本帧。示例均设为
20 ms / 20 ms / 40 ms。任一门限超出时，整个录制周期会报错，数据集不会写入这一帧。同步 sidecar
为每路保存主机采集时间、state 锚点、单路偏差、全体相机的配对偏差；Photon 另有 SDK 时间戳，便于
离线审核。

这是有上限的软件配对：Photon 的主机时间包括 USB/SDK 传输，普通 RGB 相机的时间是帧到达主机的
时刻，因此不能等同于所有传感器曝光时刻完全一致。要达到硬件级曝光同步仍需外部触发。视频编码遵循
现有 LeRobot 设置，不保证无损；`marker_motion_3d` 以数值张量写入数据集，不经过视频编码。

本次已用模拟 SDK 验证双设备、颜色转换、超时和资源释放，仍需使用真实 Photon
验证 USB 带宽、标定文件、实际帧率和 SDK 输出。
