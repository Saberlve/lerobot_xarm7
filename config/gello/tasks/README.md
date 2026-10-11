# GELLO 任务配置

每个任务一个子文件夹。配置是当前 `../xarm7_gello_base.yaml` 的独立副本；后续修改 base 不会自动同步。

共享设置与擦白板配置对齐，数据集路径、repo_id 和语言指令因任务而异；旋下螺母配置额外启用 J7 模式切换。控制 30 Hz，RGB/数据集 15 Hz，Photon 60 Hz；记录关节和 TCP（both）；关闭 GELLO 电流控制；夹爪由键盘控制，夹爪监测关闭；开启网页预览；最多录制 75 段，图像写入使用 1 个进程、每相机 1 个线程；结束录制后计算并保存 Mesh3DFlow。

| 任务 | 配置 | 数据集保存路径 | repo_id |
|---|---|---|---|
| 鸡蛋抓取 | [xarm7_gello_egg_pick.yaml](egg_pick/xarm7_gello_egg_pick.yaml) | `datasets/xarm7_gello_tasks/egg_pick` | `ufactory/xarm7_gello_egg_pick` |
| chip 抓取 | [xarm7_gello_chip_pick.yaml](chip_pick/xarm7_gello_chip_pick.yaml) | `datasets/xarm7_gello_tasks/chip_pick` | `ufactory/xarm7_gello_chip_pick` |
| 按压洗手液 | [xarm7_gello_handwash_press.yaml](handwash_press/xarm7_gello_handwash_press.yaml) | `datasets/xarm7_gello_tasks/handwash_press` | `ufactory/xarm7_gello_handwash_press` |
| 齿轮装配 | [xarm7_gello_gear_assembly.yaml](gear_assembly/xarm7_gello_gear_assembly.yaml) | `datasets/xarm7_gello_tasks/gear_assembly` | `ufactory/xarm7_gello_gear_assembly` |
| 擦白板 | [xarm7_gello_whiteboard_erase.yaml](whiteboard_erase/xarm7_gello_whiteboard_erase.yaml) | `datasets/xarm7_gello_tasks/whiteboard_erase` | `ufactory/xarm7_gello_whiteboard_erase` |
| 弹簧小车 | [xarm7_gello_spring_cart.yaml](spring_cart/xarm7_gello_spring_cart.yaml) | `datasets/xarm7_gello_tasks/spring_cart` | `ufactory/xarm7_gello_spring_cart` |
| 旋下螺母 | [xarm7_gello_nut_removal.yaml](nut_removal/xarm7_gello_nut_removal.yaml) | `datasets/xarm7_gello_tasks/nut_removal` | `ufactory/xarm7_gello_nut_removal` |
| 大号 USB 插入 | [xarm7_gello_usb_insert.yaml](usb_insert/xarm7_gello_usb_insert.yaml) | `datasets/xarm7_gello_tasks/usb_insert` | `ufactory/xarm7_gello_usb_insert` |

从项目根目录启动，例如：

```bash
.venv/bin/uf-lerobot-record --config_path config/gello/tasks/egg_pick/xarm7_gello_egg_pick.yaml
```

保存路径和电流 profile 路径均相对于启动目录。

录制中按 `←` 丢弃当前 episode 后，会暂停遥操作，先张开夹爪，再复位机械臂。
复位完成后等待 `Space` 开始重录；重新对齐 GELLO 后恢复控制，不再重复复位。
开爪失败时不执行机械臂复位；已经保存的 episode 保留。

需要拧螺丝等仅旋转末端关节的操作时，在所用任务 YAML 的 `teleop` 下显式加入：

```yaml
teleop:
  joint7_only_mode_enabled: true
```

录制和单独遥操作中，按 `S` 保持机器人当前的 J1–J6 位置，GELLO 只控制 J7；
再次按 `S` 恢复全部关节控制。恢复时 J1–J6 会按当前机器人和 GELLO 姿态重新对齐，
避免屏蔽期间移动 GELLO 后产生目标跳变。夹爪控制保持正常。
长按只切换一次；暂停时按键不切换模式，暂停、重置或下一段录制均回到全关节控制。
省略该字段或设置为 `false` 时，`S` 不生效。此模式仅支持 J1–J7 的关节空间控制。

使用 `gripper_control_mode: keyboard` 且未启用 GELLO 夹爪电流或力反馈时，
仅连接并读取 GELLO J1–J7，不读取 ID8；夹爪目标从 xArm 反馈对齐，仍由 C/O 控制。
使用 GELLO 夹爪或启用夹爪电流、力反馈时保留 ID8 通信，所需电机通信超时仍会停止控制。

弹簧小车只提供了任务名称，目前使用通用描述 `Manipulate the spring-loaded cart.`，具体运动方向和终止条件需补充。USB 插入配置以完全插入并短暂保持为结束条件。

所有任务沿用 base 的 `min_tcp_z_mm: 2`，实际接触高度需按各任务布置确认。鸡蛋配置同时适用于生鸡蛋和熟鸡蛋；如需分别统计，可复制配置并为两组设置不同的数据集路径和 repo_id。

旋下螺母配置启用 `defer_processing: true`：每条只提交完整原始图像、动作和时间索引，随后可开始下一条；退出整场录制后统一计算 Mesh3DFlow 和编码视频，录制期间不后台推理。后处理失败时原始 checkpoint 保留，可使用原配置加 `--postprocess-only` 重试。
