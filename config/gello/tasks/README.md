# GELLO 任务配置

每个任务一个子文件夹。配置是当前 `../xarm7_gello_base.yaml` 的独立副本；后续修改 base 不会自动同步。

共享设置与擦白板配置对齐，数据集路径、repo_id 和语言指令因任务而异；旋下螺母和拧紧螺母配置额外启用 J7 与可调倍率微调模式切换。控制 30 Hz，RGB/数据集 15 Hz，Photon 60 Hz；记录关节和 TCP（both）；关闭 GELLO 电流控制；夹爪由键盘控制，夹爪监测关闭；开启网页预览；最多录制 75 段，图像写入使用 1 个进程、每相机 1 个线程；结束录制后计算并保存 Mesh3DFlow。

| 任务 | 配置 | 数据集保存路径 | repo_id |
|---|---|---|---|
| 鸡蛋抓取 | [xarm7_gello_egg_pick.yaml](egg_pick/xarm7_gello_egg_pick.yaml) | `datasets/xarm7_gello_tasks/egg_pick` | `ufactory/xarm7_gello_egg_pick` |
| chip 抓取 | [xarm7_gello_chip_pick.yaml](chip_pick/xarm7_gello_chip_pick.yaml) | `datasets/xarm7_gello_tasks/chip_pick` | `ufactory/xarm7_gello_chip_pick` |
| 按压洗手液 | [xarm7_gello_handwash_press.yaml](handwash_press/xarm7_gello_handwash_press.yaml) | `datasets/xarm7_gello_tasks/handwash_press` | `ufactory/xarm7_gello_handwash_press` |
| 齿轮装配 | [xarm7_gello_gear_assembly.yaml](gear_assembly/xarm7_gello_gear_assembly.yaml) | `datasets/xarm7_gello_tasks/gear_assembly` | `ufactory/xarm7_gello_gear_assembly` |
| 擦白板 | [xarm7_gello_whiteboard_erase.yaml](whiteboard_erase/xarm7_gello_whiteboard_erase.yaml) | `datasets/xarm7_gello_tasks/whiteboard_erase` | `ufactory/xarm7_gello_whiteboard_erase` |
| 回力小车 | [xarm7_gello_pullback_car.yaml](pullback_car/xarm7_gello_pullback_car.yaml) | `datasets/xarm7_gello_tasks/pullback_car` | `ufactory/xarm7_gello_pullback_car` |
| 旋下螺母 | [xarm7_gello_nut_removal.yaml](nut_removal/xarm7_gello_nut_removal.yaml) | `datasets/xarm7_gello_tasks/nut_removal` | `ufactory/xarm7_gello_nut_removal` |
| 拧紧螺母 | [xarm7_gello_nut_tighten.yaml](nut_tighten/xarm7_gello_nut_tighten.yaml) | `datasets/xarm7_gello_tasks/nut_tighten` | `ufactory/xarm7_gello_nut_tighten` |
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

回力小车指令为 `Pull back the toy car and release it to cross the finish line.`（后拉回力小车，释放使其越过终点线）。终点线长 60 cm、宽 3 cm；以车头越过靠近起点的一侧边缘为成功判定基准。USB 插入配置以完全插入并短暂保持为结束条件。

所有任务沿用 base 的 `min_tcp_z_mm: 2`，实际接触高度需按各任务布置确认。鸡蛋配置同时适用于生鸡蛋和熟鸡蛋；如需分别统计，可复制配置并为两组设置不同的数据集路径和 repo_id。

旋下螺母配置启用 `defer_processing: true`：每条只提交完整原始图像、动作和时间索引，随后可开始下一条；退出整场录制后统一计算 Mesh3DFlow 和编码视频，录制期间不后台推理。后处理失败时原始 checkpoint 保留，可使用原配置加 `--postprocess-only` 重试。


旋下螺母 `nut_removal` 和拧紧螺母 `nut_tighten` 已设置
`teleop.joint_precision_mode_enabled: true`。开始遥操作后，按 `Z` 启用关节微调：
J1–J6 相对于切换时的角度变化乘以配置的 `joint_precision_scale`（默认 0.2，即 1/5），J7 和夹爪保持原有映射；再次按 `Z`
恢复正常倍率。两次切换均从机器人当前 J1–J6 位置重新对齐，避免目标跳变。
`S` 锁定优先，锁定期间可用 `Z` 选择退出 S 后的倍率，保持的 J1–J6 不变。
长按 Z 只切换一次；暂停时不切换，暂停、重置或下一段录制恢复正常倍率。
此功能仅支持七关节的关节空间控制，不能与 XY 平面模式同时配置。
配置启用后，网页录制工作台同样接受 `Z` 键和“切换为关节精调”按钮，并在运动模式处显示当前倍率状态；也可在启动会话前用“允许关节精调”开关做会话级控制。

在任务 YAML 的 `teleop` 下调整倍率（范围为 `0 < joint_precision_scale <= 1`）：

```yaml
teleop:
  joint_precision_mode_enabled: true
  joint_precision_scale: 0.2  # 0.2 缩小5倍，0.1 缩小10倍，0.5 缩小2倍
```

### 螺母任务：键盘 J7

`nut_removal`、`nut_tighten` 均开启 `teleop.joint7_keyboard_mode_enabled`。

- **拆卸 `nut_removal`**：普通 ↑ / ↓ 每次触发 ±150°，目标按 75°/秒推进，约 2 秒到达终点；之后继续保持终点目标，直到实时缓存中的实测 J7 与目标相差不超过 0.5°，才自动打开夹爪到 50（最大 84），夹爪速度沿用 `robot.gripper_speed`。**Ctrl + ↑ / ↓** 转 ±10°，**Shift + ↑ / ↓** 转 ±60°，同样等待实测到位，但均不自动张爪。未到位时保持忙碌、不接受下一次旋转；反馈过期按现有控制错误处理，不会因计时结束而张爪。复用已有实时状态缓存，不增加设备查询。组合键在按下方向键时确定，同时按 Ctrl 和 Shift 时 Ctrl 优先。网页提供对应按钮和快捷键。
- **拧紧 `nut_tighten`**：速度 60°/秒，配置 `joint7_keyboard_hold_to_rotate: true`。按住 ↑ / ↓ 持续正转 / 反转，松开停止并保持当前实测 J7 角度，不自动张开夹爪。持续按住可超过 120°，目标限制在 J7 关节限位内；同时按下两个方向则停止。

首次操作锁定 J1–J6；按 **S** 恢复 GELLO，并重新对齐全部七个关节。暂停、复位、断线清除控制状态。
网页按钮和上下键遵循相同模式：拧紧任务按住旋转，松开按钮、页面失焦或释放按键时停止；输入框内的上下键不触发旋转。
仅 XY 平面模式不能与键盘 J7 模式同时启用。旋转通过现有控制、限位/安全检查和数据记录链路执行。
