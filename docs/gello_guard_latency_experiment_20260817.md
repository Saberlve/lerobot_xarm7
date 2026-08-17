# GELLO 安全高度 Guard 延迟实验记录

## 1. 实验目的

GELLO 以约 60 Hz 遥操作 xArm7 时，未启用安全高度保护的运动比较平滑；加入 TCP 最低高度保护后，机械臂出现卡顿、抖动和短暂停顿。

本实验验证以下假设：

> 应用侧 guard 在控制循环中同步调用 xArm 控制器的 FK、IK 或 joint-limit 接口，导致 ServoJ 命令到达控制器的时间不均匀。

实验重点不是只看整个循环是否超过 16.667 ms，还要分别测量：

- GELLO 读取耗时
- 安全检查耗时
- ServoJ SDK 调用耗时
- 完整 `send_action` 耗时
- 相邻 ServoJ 命令的近似下发间隔

## 2. 实验环境

- 日期：2026-08-17
- 时区：Asia/Shanghai
- 机械臂：xArm7
- 控制频率：60 Hz
- 目标周期：16.667 ms
- 控制接口：`set_servo_angle_j`，即 `joint_command_mode: 1`
- 安全高度：`min_tcp_z_mm: -2.0`
- Guard 激活余量：`tcp_z_guard_activation_margin_mm: 100.0`
- Guard 同步检查激活范围：实际 TCP z 不高于约 98 mm

相关版本：

- 原始 guard 基准提交：`14c3e798f01bee60d8a888cccc6ab5ad47437378`
- 无 guard baseline 实验提交：`4231f1b`
- Guard 延迟记录提交：`c0e950f`
- Guard 延迟实验分支：`codex/gello-guard-latency`

## 3. 实验方法

### 3.1 实验 1：无 Guard Baseline

活动控制循环只执行：

```text
GELLO get_action
    -> robot.send_action
    -> ServoJ
    -> 等待下一周期
```

不执行每帧 observation、processor、FK、安全高度检查或新增 joint-limit 检查。

运行命令：

```bash
uv run uf-robot-teleop \
  --config_path config/gello/xarm7_gello_record_config.yaml \
  --robot.enable_logs=true \
  --fps 60 \
  --experiment_1_baseline=true \
  --experiment_duration_s 60
```

采用的有效数据文件：

```text
logs/gello_experiment_1_baseline_20260817_094957.csv
```

样本数为 3571，持续约 60 秒。

### 3.2 实验 2：启用应用侧 Guard

外层控制流程与 baseline 保持一致，但 `robot.send_action` 内启用 `min_tcp_z_mm` guard。记录代码进一步拆分了 safety guard 和 ServoJ 的耗时。

运行命令：

```bash
uv run uf-robot-teleop \
  --config_path config/gello/xarm7_gello_record_config.yaml \
  --fps 60 \
  --guard_latency_experiment=true \
  --experiment_duration_s 60
```

数据文件：

```text
logs/gello_guard_latency_20260817_112752.csv
```

样本数为 3570，持续约 60 秒。

每帧通过 `guard_path` 记录实际执行路径：

- `rt_fast_path`：TCP 远离高度下限，只读取异步 RT-report 状态，不执行同步 FK。
- `fk_safe`：TCP 进入 guard 激活范围，同步执行 FK 和 joint-limit 检查，目标仍安全。
- `fk_ik_clamp`：目标低于安全高度，执行 FK、IK、joint-limit 和验证 FK。
- `fallback`：安全检查失败，保持上一安全目标。

## 4. 实验结果

### 4.1 整体结果

表中数值依次为 p50 / p95 / p99 / 最大值，单位均为 ms。

| 指标 | 无 Guard Baseline | Guard 整体 |
| --- | ---: | ---: |
| 控制周期 | 16.744 / 16.929 / 17.465 / 20.594 | 16.741 / 17.013 / 17.937 / 20.165 |
| GELLO 读取 | 0.052 / 0.090 / 0.125 / 0.362 | 0.053 / 0.102 / 0.131 / 0.639 |
| Safety guard | 不适用 | 0.014 / 0.497 / 2.062 / 12.011 |
| ServoJ | 未单独记录 | 0.333 / 1.153 / 2.254 / 4.726 |
| `send_action` | 0.436 / 1.205 / 2.274 / 6.038 | 0.377 / 1.519 / 3.327 / 13.149 |
| 循环工作耗时 | 0.497 / 1.243 / 2.337 / 6.124 | 0.440 / 1.581 / 3.370 / 13.180 |

两次实验的循环工作耗时都没有超过 16.667 ms：

- Baseline：0 / 3571 帧超期
- Guard：0 / 3570 帧超期

因此，只检查“循环工作是否超过 deadline”会得到不完整的结论。

### 4.2 按 Guard 路径分组

| 路径 | 帧数 | 占比 | Guard p50 / p95 / p99 / 最大 | `send_action` p50 / p95 / p99 / 最大 |
| --- | ---: | ---: | ---: | ---: |
| `rt_fast_path` | 3324 | 93.11% | 0.014 / 0.021 / 0.031 / 0.325 | 0.363 / 1.164 / 2.255 / 4.750 |
| `fk_safe` | 246 | 6.89% | 0.689 / 3.043 / 5.283 / 12.011 | 1.002 / 4.202 / 6.715 / 13.149 |

`rt_fast_path` 与无 guard baseline 基本一致。`fk_safe` 的 `send_action` 延迟明显增大：

- p95 从 1.205 ms 增加到 4.202 ms，约为 baseline 的 3.49 倍。
- p99 从 2.274 ms 增加到 6.715 ms，约为 baseline 的 2.95 倍。
- 最大值从 6.038 ms 增加到 13.149 ms。

本次实验没有出现 `fk_ik_clamp` 或 `fallback`。因此实验期间 guard 没有使用 IK 改写目标，也没有拒绝目标；观测到的额外延迟可以单独归因于同步 FK 和 joint-limit 检查。

### 4.3 ServoJ 近似下发间隔

循环起始周期稳定并不代表 ServoJ 实际到达控制器的时间稳定。同步 FK 位于每帧 ServoJ 之前，因此 FK 耗时变化会改变 ServoJ 在该帧中的下发相位。

本实验使用以下公式推导 ServoJ 调用开始时间：

```text
近似 ServoJ 下发时刻
  = elapsed_s * 1000
  + gello_read_ms
  + safety_guard_ms
```

该值没有包含 guard 返回后到 SDK 调用前的少量 Python 开销，所以是近似值；这些开销远小于观测到的 FK 长尾，不影响结论。

| 路径 | p1 | p5 | p50 | p95 | p99 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 16.685 | 16.712 | 16.749 | 16.929 | 17.461 | 16.545 | 20.594 |
| `rt_fast_path` | 16.677 | 16.706 | 16.742 | 16.996 | 17.834 | 16.254 | 20.171 |
| `fk_safe` | 12.751 | 14.916 | 16.760 | 19.093 | 20.825 | 6.756 | 28.817 |

`fk_safe` 中出现了典型的长短周期交替：

```text
28.817 ms -> 6.756 ms
25.510 ms -> 7.243 ms
20.877 ms -> 12.723 ms
20.756 ms -> 12.787 ms
```

其表现是控制器先较长时间收不到新目标，随后在很短间隔内收到下一目标，符合实际体感中的“停一下，然后突然追赶”。

### 4.4 Guard 激活时间段

本次实验中同步 FK 路径集中在以下时间段：

| 路径 | 帧范围 | 时间范围 | 帧数 |
| --- | ---: | ---: | ---: |
| `rt_fast_path` | 0-2352 | 0.000-39.516 s | 2353 |
| `fk_safe` | 2353-2398 | 39.533-40.288 s | 46 |
| `rt_fast_path` | 2399-3369 | 40.304-56.610 s | 971 |
| `fk_safe` | 3370-3569 | 56.627-59.984 s | 200 |

若机械臂的卡顿体感集中在约 39.5-40.3 秒和 56.6-60.0 秒，则与同步 FK 路径在时间上直接吻合。

### 4.5 其他错误排查

`logs/xarm7_gripper_errors.log` 中本次实验之前最近的 controller error 时间为 11:20。本次 guard 实验约在 11:26-11:27 运行，期间没有新增 gripper/controller error，因此这些错误不是本次延迟长尾的原因。

## 5. 结论

实验结果支持最初假设：

1. GELLO 读取非常快，不是卡顿来源。
2. ServoJ SDK 调用存在少量长尾，但 baseline 和 `rt_fast_path` 表现相近，不是加入 guard 后才出现的主要变化。
3. TCP 远离安全高度时，RT-report 快速路径几乎没有额外成本。
4. TCP 接近安全高度后，同步 FK 和 joint-limit 检查产生 3-12 ms 的不稳定延迟。
5. 即使整个循环工作耗时仍小于 16.667 ms，同步检查也会改变 ServoJ 在帧内的下发相位，造成约 6.8-28.8 ms 的不均匀命令间隔。
6. 本轮没有 IK clamp 或 fallback，因此不需要用“关节目标被修改”来解释抖动；仅同步控制器通信已经足以解释现象。

综合判断：应用侧同步安全检查是加入 guard 后抖动的主要原因。

## 6. 后续方案

建议按以下优先级处理：

1. 优先使用 xArm 控制器侧 TCP 安全边界，让控制器在内部阻止越界，不在 60 Hz Python 发送线程中同步查询 FK/IK。
2. 如果必须在应用侧检查，使用本地运动学库计算 FK/IK，避免每帧和控制器进行同步请求。
3. 如果本地计算不可用，将安全计算与 ServoJ 发送线程解耦，并采用保守的最后安全目标策略；需要另外验证异步结果的时效性和安全性。
4. 保留 controller error、guard path 和各阶段延迟记录，后续方案必须用相同实验复测。

不建议为了平滑性直接移除实际运行中的安全保护。无 guard 模式仅用于受控环境下建立 baseline。

## 7. 下一轮验收指标

控制器侧安全边界方案应至少满足：

- `send_action` p95/p99 接近 baseline。
- ServoJ 下发间隔不再因接近安全高度出现明显长短周期交替。
- 靠近安全面和触发安全边界时均不执行 Python 侧同步 FK/IK。
- 保持 TCP 最低高度保护有效。
- 无新增 controller、gripper 或通信错误。
