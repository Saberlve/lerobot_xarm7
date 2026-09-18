# xArm7 → GELLO ID1–7 实验性关节反馈

默认关闭。软件测试不代表真机可用性验证；真实采样率、串口吞吐量、方向、外力估计精度均须按下面阶段验证。没有改动 FACTR 仓库、TCP 高度保护、位置映射或 ID8 信号处理公式。

## 1. Repository audit

- FACTR `factr_teleop.py:torque_feedback()` 提供外力矩比例反馈和 leader 速度阻尼；`control_loop_callback()` 组合限位、零空间、重力、摩擦和反馈。
- `set_leader_joint_torque()` 最后还乘 joint_signs；不能只复制 feedback 内的负号。Franka 示例的反馈阻尼为 0。
- `factr_teleop_franka_zmq.py` 取 ZMQ 最新缓存，没有样本年龄检测。`shut_down()` 有力矩关闭，但一般异常的退出覆盖不足。
- FACTR 的 `dynamixel/driver.py` 使用地址 102、两字节有符号电流及 GroupSyncWrite，按型号转换，统一 raw ±900 限幅。本实现只借鉴控制结构和同步写方式，不复制增益、motor_scalar、符号、限幅或 leader URDF。
- 当前 `uf_robot.get_observation()` 调用 SDK `get_joint_states(num=3)`，只使用位置及可选速度；没有外力估计。
- 当前 ID8 链路：`RealtimeTeleopController` → `GripperFeedbackProcessor` → GELLO latest target → ID8 worker → `SafeDynamixelDriver`。读/写使用同一驱动锁。
- 本地 GELLO 包已有位置/速度 SyncRead 后台线程，但缺少样本时间戳。新增 timed reader 只在配置启用 arm feedback 时使用；默认仍运行原 reader。
- 已有 `local_kinematics.py` 是毫米制运动学模块，不是动力学模型；TCP target projection 保持原样。

## 2. xArm feedback source

核对的本地 SDK 版本：**1.18.4**。

| 信号 | 定义与选择 |
|---|---|
| `get_joint_states(is_radian=True,num=3)` | 返回 position、velocity、effort。源码 `xarm/x3/base.py` 直接将 `ret[15:22]` 作为 effort，没有外力估计或单位转换。默认选择该请求接口，以便每次成功请求都有保守时间戳。 |
| `get_joints_torque()` / `joints_torque` | SDK 叫 joint torque，没有在该接口保证已经去除重力、惯性、摩擦。第一版不据此假定 external torque。 |
| `currents` | 独立的伺服电流属性，不能直接作为关节外力矩；本版不将其接入 active。 |
| `get_ft_sensor_data(is_raw=False)` | 官方六维 FT 的滤波、负载/偏置补偿结果；需受支持传感器和固件。可选 `ft_sensor` 源。 |
| `ft_ext_force` | SDK 的缓存属性；没有独立样本时间戳，所以本版优先 FT 请求接口。 |

参考：[官方 SDK API](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/doc/api/xarm_api.md)、[SDK base 实现](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/xarm/x3/base.py)。

默认 `bias_compensated_joint_effort`：

```text
raw_joint_effort [sdk_effort_unit]
  - 固定 baseline[7]
= estimated_contact_torque [sdk_effort_unit，非已确认 Nm]
```

这是同姿态/小姿态范围的准静态实验估计。baseline 不会在线更新，因此不会把持续接触自动滤掉；但姿态变化、摩擦和惯性仍会产生假接触信号。`raw_joint_effort` 源只允许 observe-only。`disabled` 不启动采样。

SDK 文档未明确 effort 的物理单位，因此记录为 `sdk_effort_unit`，而不是虚构 N·m。增益叫 `gain_ma_per_unit`，表示 **mA/SDK effort unit**。若选择 FT，该增益单位变为 mA/N·m。没有用电机堵转力矩伪造精确力矩常数。

FT 使用同一次采样过程中的 q 和 wrench；两次 RPC 不是硬件同步，误差需实测。`ft_sensor_to_flange` 必须显式配置传感器坐标到法兰坐标的 4×4 刚体变换（平移 mm），实际传感器输出坐标也必须匹配。使用校准法兰运动学，在传感器原点计算数值几何雅可比；力和力矩旋转到模型世界坐标。平移雅可比除以 **1000** 后再乘 N，旋转部分直接乘 N·m。`ft_vertical_only` 先旋转到世界坐标，再取世界 Fz，并非直接取传感器 Fz。

FT 接触估计覆盖传感器负载路径上的工具接触，不能代替整臂碰撞检测。没有自动传感器调零、负载辨识或动力学补偿。

## 3. Architecture

```text
独立只读 xArm SDK connection
  get_joint_states / 可选 get_ft_sensor_data
          ↓ sampler，单槽 latest sample（无 force queue）
  固定基线 / FT JᵀW
          ↓
  ArmFeedbackProcessor：有效性 → bias → clamp → signed deadzone
    → EMA（仅新样本）→ 每轴 gain/sign → leader velocity damping
    → 按真实 dt 的 slew → 每轴 mA clamp → enabled_joints mask
          ↓
  ArmFeedbackWorker + 独立软件 watchdog + CSV/JSON
          ↓
  GelloArmFeedbackAdapter → shared driver._lock → SyncWrite ID1–7

原 G2 report → 原 GripperFeedbackProcessor → 原 ID8 worker
          ↓
  原 ID8 adapter → 同一个 driver._lock → ID8 Goal Current

原 GELLO 位置 → 原 realtime position loop → 原 TCP guard → xArm
```

采用共享锁的最小修改方案，而非重写 ID8 为新 single writer。ID1–7 每次只做一个 GroupSyncWrite；ID8 保留原有语义。初始化、健康检查和清理使用带应答的逐电机访问；它们不是逐电机的周期电流输出循环。

## 4. Files changed

| 文件 | 职责 |
|---|---|
| `utils/arm_feedback.py` | 七维配置验证、样本/结果、无 I/O 处理器、显式单位的 FT 映射 |
| `utils/arm_feedback_runtime.py` | 独立采样连接、latest sample、worker/watchdog、CSV 和 metadata/status |
| `teleoperators/gello_teleop/arm_adapter.py` | 型号探测、显式电流会话、SyncWrite、回滚、温度/硬件错误检查、timed reader |
| `gello_adapter.py` | 选择 opt-in reader、禁止 active 时写位置/全局启动力矩、关闭会话；原 ID8-only 保护保留 |
| `gello_teleop_config.py` | 嵌套 `arm_feedback` dataclass，默认 disabled |
| `gello_teleop.py` | 启停机械臂反馈，在 pause/reset/disconnect 清理 |
| `utils/realtime_teleop.py` | ID8 初始化后启动机械臂反馈；失败记录日志并保留位置/ID8 遥操作；退出分别清理 |
| `scripts/uf_test_arm_force_feedback.py` | 无 follower 运动命令的诊断入口、CSV 汇总、CPU benchmark |
| `config/arm_feedback/observe.yaml` | 与完整 GELLO 配置分离的实验模板 |
| `tests/test_arm_feedback*.py` | 处理器、驱动、线程/日志/配置/双通道回归 |
| `pyproject.toml` | 注册 `uf-test-arm-force-feedback` |

## 5. Safety

- 所有七维参数验证 shape、finite、范围；符号只接受 ±1，mask 只接受 bool。
- active 必须显式启用，选择关节，并确认 baseline 和 sign；原始 effort 模式拒绝 active。默认 gain 为零、所有关节关闭；实验模板上限为 5 mA、slew 为 5 mA/s。
- 启动探测所有 ID1–7，未知型号在任何电流模式写入之前拒绝。只对 selected joints 切模式/启用力矩。
- 当前支持：XL330-M077-T (1190，1 mA/raw)、XL330-M288-T (1200，1 mA/raw)、XC330-T288-T (1220，1 mA/raw)、XM430-W210 (1030，2.69 mA/raw)。实际型号未测量，写入前必须由 ping 返回支持表中的型号。
- 硬件 Current Limit 必须有效；软件上限不能超过硬件限值，并额外限制为 ≤100 mA。**100 mA 是实验软件上限，不是所有机械结构的安全认证值。** 不写 EEPROM Current Limit，不自动提高已有硬件限制。
- 模式序列：torque off → current mode → verify → Goal Current=0 → verify → torque on → verify。不能把“启用后再归零”作为启动顺序。
- 第一个处理输出为零。正常限幅/斜率限制在 mA 域；转换为寄存器后再裁剪。量化会产生 1 raw 的离散台阶，所以实际寄存器电流不能严格呈连续斜坡；日志区分 hypothetical 与量化后的 command。
- xArm 样本 stale、leader stale、SDK 错误、NaN/Inf、处理超时/异常、write error 会锁存 fault 并尽力 zero/disable。故障清零绕过 slew，禁止自动恢复非零电流。
- 获取串口锁后再次检查 command deadline，过期目标不发送。
- 独立软件 watchdog 在输出/日志线程停滞时尝试停用。健康检查每约 0.5 秒读取 selected motors 的温度及 Hardware Error Status；阈值默认 55°C，配置不得超过 60°C。
- shutdown 和初始化回滚逐电机尝试 zero、torque off、恢复模式、再次 zero；一个操作失败仍继续剩余操作。不恢复先前 torque-on 状态。
- 软件 watchdog 与硬件使用同一串口。如果进程被强杀、主机挂死、串口永远阻塞或线路断开，不能保证零电流命令送达。没有把持续 SyncRead 下的 Bus Watchdog 当作非零电流超时保护。
- sign_product 只是诊断，不证明稳定性；阻尼使用 leader 速度，并按电机坐标方向转换，独立于接触反馈 sign。

型号来源：[XL330-M077](https://emanual.robotis.com/docs/en/dxl/x/xl330-m077/)、[XL330-M288](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/)、[XC330-T288](https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/)、[XM430-W210](https://emanual.robotis.com/docs/en/dxl/x/xm430-w210/)。

## 6. Realtime design

不改变原 position control fps。目标采样/输出默认各 100 Hz，可显式提高到 200 Hz，但不保证达到。

有独立 xArm sampler、haptic worker、watchdog；Dynamixel 沿用一个后台 SyncRead，ID8 沿用自己的输出 worker。全部串口访问共用一把锁，机械臂处理线程读取不可变的最新状态快照。

时间使用 monotonic_ns。xArm 取请求开始时间作为保守样本时间；leader 取 SyncRead 请求开始时间，避免给慢响应重新盖“新鲜”时间戳。记录 sequence、period、age、读延迟、处理延迟、写调用延迟（含锁等待/健康检查）、纯 SyncWrite transaction 时间、loop dt 和 overrun。

CSV 每个 haptic tick 一行；若采样更快，记录的是被消费的最新样本，period 分位数来自这些样本。平均采样/读取 Hz 使用 sequence 跨度计算，避免把重复消费计为新样本。API 响应频率不是传感器内部采样频率。sample→command 是主机请求开始至写调用返回的延迟，不是电机真实力矩生效时间。

本地上游 GELLO 驱动默认波特率为 57600；本实现不自动修改电机波特率或 USB 配置。低波特率和逐电机健康检查可能导致超时安全退出。应先看 Stage A 实测数据，再统一手动配置总线和合理的超时/速率；不能只增大 timeout 掩盖延迟。

## 7. Tests and benchmark

在仓库根目录（Windows 本地环境）：

```powershell
.venv\Scripts\python.exe -m pytest -o pythonpath=src -p no:cacheprovider -q
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m lerobot_robot_ufactory.scripts.uf_test_arm_force_feedback --benchmark
```

CPU-only 10,000 次处理器测量，200 次预热（本次 Windows / Python 3.13）：p50 **0.0446 ms**，p95 **0.0488 ms**。这是本次软件结果，不等于真实 haptic 周期。

本次完整软件测试：**246 passed**，其中新增机械臂测试 56 项。新增 Python 文件 Ruff 检查通过，`git diff --check` 通过。没有连接或驱动真机。

| 真机指标 | 本次结果 |
|---|---|
| xArm sampling Hz / period p50,p95 | NOT MEASURED |
| GELLO read Hz | NOT MEASURED |
| Dynamixel write p50,p95 / Hz | NOT MEASURED |
| sample→current command p50,p95 | NOT MEASURED |
| 真机 stale / write error / overrun count | NOT MEASURED |

## 8. Stage A — Observe-only command

以下命令在项目已有机器人运行环境执行，`python` 应指向安装了项目依赖的解释器。Windows 本仓库可先设置 `PYTHONPATH=src`，并将 `python` 换成 `.venv\Scripts\python.exe`。

```text
python -m lerobot_robot_ufactory.scripts.uf_test_arm_force_feedback --config-path config/gello/xarm7_gello_test.yaml --feedback-config config/arm_feedback/observe.yaml --joints 1 2 3 4 5 6 7 --duration 60 --log-path logs/arm_stage_a_free.csv
```

该工具只从原配置读取 IP、GELLO port 和关节设置；不连接相机，不调用 xArm motion_enable、set_mode、set_state 或发送位置，**不会自动让 xArm 跟随 GELLO**。通过已有的机器人操作方式进入合适的手动/示教状态，先记录同姿态无接触，再另一次轻微接触。它不驱动 ID8；需要 ID8 同时工作时使用下文的正常遥操作接入，而非两个进程抢同一串口。

```text
python -m lerobot_robot_ufactory.scripts.uf_test_arm_force_feedback --config-path config/gello/xarm7_gello_test.yaml --feedback-config config/arm_feedback/observe.yaml --joints 1 2 3 4 5 6 7 --duration 60 --log-path logs/arm_stage_a_contact.csv
python -m lerobot_robot_ufactory.scripts.uf_test_arm_force_feedback --summarize logs/arm_stage_a_free.csv
```

先比较自由空间、同姿态接触、离开接触，以及小范围姿态变化。只对明确无接触、近静止片段计算 baseline。汇总中的 raw_effort_median 只是建议值，包含接触的整段 CSV 不能用作 baseline。

## 9. Stage B — Single-joint low current

将 `config/arm_feedback/observe.yaml` 复制为 `config/arm_feedback/calibrated.yaml`。填入实测 baseline、噪声死区及选定关节的 sign。完成基线和方向约定核对后，显式设置 `baseline_verified: true`、`sign_verified: true`。这些是操作者的校准声明，软件不能代替真机方向验证。

保留 5 mA 上限、5 mA/s slew、零 damping。示例用 ID7，但不声称它对所有机械结构都是最安全关节；应由实际结构决定。首测仅短时运行并保持能立即 Ctrl+C/切断电机供电。

```text
python -m lerobot_robot_ufactory.scripts.uf_test_arm_force_feedback --config-path config/gello/xarm7_gello_test.yaml --feedback-config config/arm_feedback/calibrated.yaml --active --joints 7 --duration 10 --log-path logs/arm_stage_b_j7.csv
```

未经填写校准声明，active 命令会在硬件连接前拒绝执行。接触时应产生预期阻力；若出现助推立即停止，修改 sign 后重新短测。不得在电机持续输出时修改配置。量化后电流可能很小或为零，先检查日志，不要直接大幅提高增益。

## 10. Stage C — Full seven-joint command

先分别验证各关节，再验证 2–3 个关节。只有这次所有 selected joints 的方向和基线已核对，才执行：

```text
python -m lerobot_robot_ufactory.scripts.uf_test_arm_force_feedback --config-path config/gello/xarm7_gello_test.yaml --feedback-config config/arm_feedback/calibrated.yaml --active --joints 1 2 3 4 5 6 7 --duration 30 --log-path logs/arm_stage_c.csv
```

逐项调 deadzone、gain、EMA、damping、slew 和 current limit。不要从 FACTR 参数起步。暂停/故障后不会自动恢复力反馈。

### 与原位置遥操作、ID8 同时运行

复制原完整 `config/gello/xarm7_gello_test.yaml`（或当前实际使用的完整配置）为新文件。在 `teleop` 下新增 `arm_feedback` 映射，使用已校准参数，设置 `enabled: true`、`observe_only: false` 和需要启用的 mask。原 robot/TCP 保护和全部 ID8 字段保持原值。然后：

```text
python -m lerobot_robot_ufactory.scripts.uf_robot_teleop --config_path=config/gello/xarm7_gello_arm_calibrated.yaml
```

该接入使用同一个 GELLO 实例/串口锁，不要并行运行诊断工具。ID8 开关独立；`arm_feedback.enabled: false` 时继续原夹爪反馈。暂停、重置、关闭实时控制器或 disconnect 都会停止机械臂反馈；恢复由新的实时控制会话显式启动。机械臂启动失败会报错并保留原位置/ID8 遥操作，不会假装反馈已生效。

## 11. Logs

每个日志路径对应：

- `.csv`：timestamp、七轴 raw/baseline/estimate、leader position/velocity、processed/hypothetical/actual current、sign product、时序、stale/clamped/fault。
- `.metadata.json`：参数、启动读取的型号、硬件电流限值及单位。
- `.status.json`：锁存 fault、stale/write-error/overrun 计数，包括未能写进 CSV 的 worker 异常。

同名 CSV 已存在时自动增加时间戳后缀，不覆盖历史，控制台显示实际路径。`command_current_ma` 在 observe-only 中全部为零；`hypothetical_current_ma` 是未下发的连续 mA 目标；active 的 command 为量化后的请求值，**不是实测电流**。

```text
python -m lerobot_robot_ufactory.scripts.uf_test_arm_force_feedback --summarize logs/arm_stage_c.csv
```

回传 Stage A 的 free/contact CSV、Stage B/C CSV、各自 metadata/status JSON 和故障终端输出。不要只回传最后一个电流值。

## 12. Remaining risks

effort 单位和真实外力语义未由硬件验证；固定基线不补偿姿态、工具负载、惯性与摩擦。实际电机型号、力矩常数、反馈方向、电流承载能力和温升未实测。FT 依赖正确的传感器原点/坐标、单位、负载补偿及两次 RPC 的时间差。软件不是硬实时系统，锁竞争、USB 延迟、日志写盘和 Python 调度都可能触发保护。广播 SyncWrite 返回成功只说明主机发包成功，不确认所有电机实际执行。

重力补偿、零空间力矩、关节力矩驱动的 follower 自动 hold/retreat 以及硬件故障下保证断力，都不在本版实现范围。已有 TCP 几何保护继续独立生效。
