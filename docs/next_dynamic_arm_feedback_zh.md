# xArm7 NEXT 动态外力矩估计集成

本文记录 `factr2_next` 到本项目的代码审计、信号映射、运行命令和 Stage C
实机验证顺序。NEXT 只替换 external-torque estimator；现有 sign、gain、滤波、
slew、current clamp、GELLO Current Mode 和 ID8 夹爪反馈均继续使用原实现。

## 1. 最终链路

```text
xArm rich report (同一快照)
  q [rad] + qdot [rad/s] + measured effort [SDK effort unit]
                         +
  safety-checked follower q_cmd [rad]
                         |
                         v
       NEXT [q, qdot, q_cmd - q] rolling history
                         |
                  tau_free_pred
                         |
       tau_ext_raw = tau_measured - tau_free_pred
                         |
       bias -> deadzone -> EMA -> contact hysteresis/ramp
                         |
       calibrated sign/gain -> leader velocity damping
                         |
                 slew limit -> mA clamp
                         |
          GELLO ID1--7 Goal Current (address 102)

G2 measured current -> existing GripperFeedbackProcessor -> GELLO ID8
```

`joints_torque` 的 xArm SDK 1.18.4 单位未在当前硬件链路中得到可靠文档确认，
因此代码不会把它错误标为 N-m。NEXT 的预测值、残差、baseline、deadzone 和 contact
threshold 始终使用训练 target 的同一单位。只有在数据源经独立校准确认是 N-m 后，
`threshold_nm` 才能按字面解释为 N-m。

## 2. `lerobot_xarm7` 审计

| 项目 | 结论 |
|---|---|
| q | `XArmFeedbackSource._report_snapshot()` 读取 rich-report `api.angles` |
| qdot | 同一 packet 的 `api.realtime_joint_speeds` |
| q_cmd | `robot.send_action()` 返回的 safety-checked effective action，经线程安全快照传给 worker |
| effort | rich-report cache `api.joints_torque`；不用可能冻结的 `get_joint_states(num=3)` 值 |
| 单位 | q 为 rad，qdot 为 rad/s，q_cmd 为 rad；effort 保持 SDK 原始单位 |
| joint order | 固定 `J1, J2, ..., J7`；GELLO arm ID 固定 `1..7` |
| 控制频率 | GELLO→xArm 默认 30 Hz；xArm sampler 配置 20 Hz；反馈 worker 配置 100 Hz，但不会重复滤波同一 sample |
| GELLO 写入 | `GelloArmFeedbackAdapter` 对选中 ID1--7 做 address 102 GroupSyncWrite |
| static feedback | `XArmFeedbackSource` baseline residual + `ArmFeedbackProcessor` |
| dynamic feedback | `ArmFeedbackWorker` + `ArmExternalTorqueEstimator` backend |
| YAML | `GelloTeleopConfig.arm_feedback` 解码为强类型 nested dataclass |
| 日志 | 每 session 新 CSV、metadata JSON、status JSON；不会覆盖旧实验 |
| ID8 | 独立 worker、独立 enable/zero/disable；arm adapter 明确不接触 ID8 |
| stale/disconnect | report、sample、GELLO leader、q_cmd、worker heartbeat 和 write deadline 均有 gate |

## 3. 官方 NEXT 审计与映射

官方仓库：<https://github.com/philiphan0109/factr2_next>

| 官方 NEXT | xArm7 映射 |
|---|---|
| `joint_pos` | report-synchronized `api.angles`, J1--J7 |
| `joint_vel` | report-synchronized `api.realtime_joint_speeds`, J1--J7 |
| `joint_cmd` | follower 实际发送的安全保护后 joint target |
| feature | `[q, qdot, q_cmd - q]`, 每帧 21 维 |
| default history | 50；checkpoint 中保存并由 runtime 强制采用 |
| model | checkpoint-compatible MLP/GRU/LSTM；默认 stateless、单向、2×128 LSTM + regression head |
| normalization | 仅用 train split 拟合 x/y mean/std；runtime 完全复用 |
| target | contact-free 数据当前帧的 measured joint torque，不预测未来帧 |
| prediction | `tau_free_pred`，即无接触自由运动力矩 |
| residual | `tau_measured - tau_free_pred` |
| checkpoint | `model.pt + config.yaml + normalization.npz` |
| runtime | rolling `HistoryBuffer`；窗口未满时绝不输出非零 feedback |
| contact | residual 滤波后逐关节 hysteresis/debounce；time-based ramp |
| feedback | gate × residual × gain/sign − leader velocity damping，再 slew/clamp |

复用并保持一致的纯算法部分：HistoryBuffer feature contract、滑动窗口 target 对齐、
MLP/GRU/LSTM 架构、stateless recurrent 语义、train-only normalization、checkpoint
字段、inference preprocessing 和 output denormalization。

未迁移部分：ROS2 node/topic/message_filters、Piper driver、ament/launch、ROS web
visualizer，以及 Piper/GELLO demo 的 torque-output 实现。它们是 transport 或特定硬件
代码；当前项目直接复用已有 xArm/GELLO runtime 和安全写入路径。

## 4. 配置与 fallback

参考配置为 `config/gello/xarm7_gello_next.yaml`。关键开关：

```yaml
arm_feedback:
  estimator:
    mode: next
    shadow_baseline: true
  next:
    enabled: true
    checkpoint: runs/xarm7_next/model.pt
    normalization: runs/xarm7_next/normalization.npz
    config: runs/xarm7_next/config.yaml
    device: cpu
    fallback: baseline        # 或 disable
    inference_timeout_ms: 50
    command_stale_timeout_ms: 500
  contact:
    enabled: true
    threshold_nm: [2, 2, 2, 2, 2, 2, 2]
    release_threshold_nm: [1, 1, 1, 1, 1, 1, 1]
    debounce_ms: 30
    ramp_up_ms: 250
    ramp_down_ms: 150
```

模型 load/inference 失败会明确写入 `estimator_mode=baseline_fallback` 或
`next_disabled` 及错误文本，绝不 silent failure。history 未满时始终保持 current=0，
不会为了 fallback 绕过 warm-up。active 后 q_cmd 丢失或 stale 会清零并锁存 fault。

## 5. 数据、训练与离线评估

安装 NEXT 可选依赖：

```bash
pip install -e ".[next]"
```

采集 contact-free teleoperation 数据：

```bash
python -m lerobot_robot_ufactory.scripts.uf_record_next_training_data \
  --config_path config/gello/xarm7_gello_next.yaml \
  --output logs/next_free_motion.csv
```

此入口强制 `enabled=true, observe_only=true, dynamic_mode=true`，并在采集期间使用
baseline logging backend（训练 target 不依赖尚未产生的 NEXT checkpoint）。全程不得触碰机械臂，
应覆盖慢速、中速、加减速、反向、不同姿态和多关节耦合运动。CSV 包含 timestamp、
q/qdot/qcmd/qerror、measured torque、robot state/mode、loop/xArm latency、valid/stale
flags，以及完整 estimator/feedback diagnostics。

训练（多个文件时逐个列出，Windows PowerShell 不自动展开 glob）：

```bash
python -m lerobot_robot_ufactory.scripts.train_xarm_next \
  --data logs/free_01.csv logs/free_02.csv \
  --output runs/xarm7_next \
  --history 50 --model lstm --epochs 100 --device cpu
```

训练使用 deterministic seed、contiguous validation split 和 history-sized leakage
buffer，保存 validation loss 最低的模型，并输出 `model.pt`、`config.yaml`、
`normalization.npz`、`metrics.json`；metrics 包含每 epoch 的 train/validation loss 和
每关节物理单位 RMSE。

离线评估：

```bash
python -m lerobot_robot_ufactory.scripts.eval_xarm_next \
  --run-dir runs/xarm7_next \
  --data logs/free_validation.csv --device cpu
```

## 6. Stage C 实机验证

### C0 — offline

运行 unit tests 和 offline evaluator。确认 checkpoint input/output 是 `21/7`、joint
order 是 J1--J7、normalization finite、validation residual 可接受。

### C1 — observe-only free motion

```bash
python -m lerobot_robot_ufactory.scripts.uf_robot_teleop \
  --config_path config/gello/xarm7_gello_next.yaml
```

保持 `observe_only: true` 与所有 `enabled_joints: false`。确认 `history_ready` 后 free
motion 的 `tau_ext_raw/tau_ext_filtered` 接近 0，且 `feedback_current_ma_*` 始终为 0。

### C2 — observe-only manual contact

仍保持 observe-only。逐方向轻触，确认相应 residual 明显高于 free-motion noise，contact
enter/release 无快速抖动，ramp 连续。

### C3 — shadow comparison

保持 `shadow_baseline: true`，比较同一日志的 `tau_ext_baseline_*` 和
`tau_ext_filtered_*`。自由运动应由 NEXT 给出更低 residual；接触时应保留更高 SNR。

### C4 — single-joint active

先把该关节 gain/current limit 调到很小，再执行（示例仅 J1）：

```bash
python -m lerobot_robot_ufactory.scripts.uf_robot_teleop \
  --config_path config/gello/xarm7_gello_next.yaml \
  --teleop.arm_feedback.observe_only=false \
  --teleop.arm_feedback.enabled_joints=[true,false,false,false,false,false,false]
```

每个关节分别验证方向、release、stale、Ctrl+C 和断连清零。禁止直接从 J1 跳到全关节。

### C5 — multi-joint active

一次只增加一个已验证关节，保持保守 gain、slew 和 current limit。

### C6 — dynamic teleoperation

所有关节逐个通过后才允许：

```bash
python -m lerobot_robot_ufactory.scripts.uf_robot_teleop \
  --config_path config/gello/xarm7_gello_next.yaml \
  --teleop.arm_feedback.observe_only=false \
  --teleop.arm_feedback.enabled_joints=[true,true,true,true,true,true,true]
```

## 7. Fail-safe 清单

- ID1--7 per-joint enable mask、mA clamp、硬件 Current Limit 与 100 mA software ceiling。
- report/sample/GELLO/q_cmd stale gate；inference timeout；worker heartbeat watchdog。
- q/qdot/qcmd/torque/model output/feedback 对 NaN/Inf 拒绝；可配置 `spike_limit`
  超限立即清零并锁存 fault。
- model load/inference failure 的 YAML fallback，日志记录 mode/status。
- history invalid 时零输出；observe-only 永不 enable current mode。
- contact hysteresis、debounce、ramp；EMA、slew 和 output clamp。
- write deadline、serial exception、Dynamixel health/temperature fault立即 disable。
- startup partial failure、processing exception、Ctrl+C、finally、disconnect 都调用
  adapter disable：逐 ID 写 Goal Current=0、Torque Disable、恢复原 mode，再次写 0。
- Arm adapter 只枚举/写 ID1--7；ID8 原夹爪链路独立保留。

阈值、gain、sign 和 current limit 都属于实机标定参数。示例配置不是通用安全值。
