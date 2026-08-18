# xArm7 + GELLO：消除遥操作抖动并可靠录制数据

这篇文档记录一次真实排障。目标看起来很简单：GELLO 控制 xArm7 时保留第七关节，TCP 又不能撞桌，同时录下同步的图像、关节状态和动作。实际遇到了三个互相关联的问题：机械臂抖动、录制数据时间对不上、夹爪触发控制器 Error 19。

下面不堆术语，先讲最终答案，再解释为什么。

## 最终方案

- GELLO 始终发送七个关节角，机械臂使用 `set_servo_angle_j`，控制频率 60 Hz。这样 J7 不会因为改发 EEF 位姿而丢失。
- 60 Hz 控制放在独立实时线程里。相机读取、图片编码和数据写盘再慢，也不能拖慢关节命令。
- TCP 高度保护使用 CPU 本地运动学计算，不在控制循环里向 xArm 控制器同步请求 FK/IK。
- 控制器 Safety Boundary 仍然开启，作为本地保护之外的最后一道硬保护。
- 相机和数据集按真实的 30 Hz 记录，机械臂控制仍保持独立的 60 Hz。
- 每条 action 带发送时刻；每次 RT joint state 采样也带时刻。写入数据集时，为 state 选择当时已经发送的最近 action，绝不拿“未来的 action”配较早的 state。
- xArm Gripper 保持 60 Hz 的目标检查能力，但只有变化超过阈值才真正发送；夹爪速度从最大值 5000 降到 1500，解决 Error 19。

当前使用的配置是：

```yaml
robot:
  control_space: "joint"
  joint_command_mode: 1

  min_tcp_z_mm: 70.0
  tcp_z_guard_backend: "local_projection"
  tcp_z_soft_margin_mm: 5.0
  controller_safety_boundary: true

  gripper_command_interval_s: 0.0166667
  gripper_command_threshold: 0.01
  gripper_speed: 1500

teleop:
  realtime_control_fps: 60

dataset:
  fps: 30
```

完整配置见 `config/gello/xarm7_gello_record_config.yaml`。

## 一、遥操作为什么会抖

最初为了保护 TCP 高度，每一轮控制都做了下面这些事：

```text
读取 GELLO
  -> 请求控制器做 FK / IK / joint-limit 检查
  -> 发送 ServoJ
```

问题不在 FK 数学本身，而在“每一帧都要等控制器回复”。一次请求可能很快，下一次却慢几毫秒。60 Hz 控制周期只有 16.67 ms，这种不稳定延迟会让 ServoJ 命令变成：等得久一下，紧接着又很快补一条。机械臂体感就是停一下、追一下，也就是抖动。

延迟实验已经观测到接近安全高度时，ServoJ 间隔会在约 6.8–28.8 ms 之间长短交替。即使平均频率看起来仍是 60 Hz，命令到达时间不均匀也足以造成抖动。详细原始实验见 `docs/gello_guard_latency_experiment_20260817.md`。

录制时还有第二个抖动来源：原来的单线程循环要依次读取机器人、读取两路相机、处理图像、写数据，然后才发送下一条控制命令。相机一次等待约 33 ms，直接把 ServoJ 卡住。实验模式不读取相机所以很平滑，正式录制却抖，差别就在这里。

## 二、怎么消除抖动

### 1. 控制和录制彻底分开

新增固定频率控制器 `RealtimeTeleopController`。它只负责：

```text
每 16.67 ms：读取 GELLO -> 本地安全检查 -> 发送 ServoJ
```

主录制线程负责：

```text
读取 RT state -> 读取相机 -> 组装数据 -> 写入 dataset
```

两者互不等待。相机偶尔慢一帧，只会影响这一帧什么时候写完，不会改变 ServoJ 的节拍。主线程如果卡死超过 1 秒，控制线程会自动停止，避免程序异常后继续下发命令。

普通 `uf-robot-teleop` 也使用同一个实时控制器，所以录制结束后单独遥操作不会重新掉回容易抖动的旧循环。

### 2. 安全检查留在本机 CPU

xArm7 的本地模型从控制器读取一次标定参数，启动时与控制器 FK 做交叉验证。验证通过后，每个控制周期的 TCP 高度和雅可比投影都在 CPU 计算，不再产生控制器网络往返。

这里选择 CPU 而不是 GPU，是因为一次 7 自由度 FK/Jacobian 计算非常小。GPU 的数据搬运和调度开销反而更大；5090 对这种单样本、低维计算没有速度优势。

正常目标原样发送。目标准备穿过软高度面时，只投影会继续降低 TCP 的关节运动分量，而不是改发六维 EEF 命令，因此七个关节自由度仍然保留。控制器侧的 Safety Boundary 放在硬下限处兜底。

## 三、怎么确认录制的数据同步

把控制拆到另一个线程以后，不能简单地在相机读完后拿“最新 action”。那条 action 可能是在 state 采样之后才发送的，相当于用未来动作解释过去状态。

现在的配对规则是：

1. 从 xArm RT report 复制关节 state 时，立即记录单调时钟 `state_sample_s`。
2. 每次 ServoJ 实际发送完成后，记录 action 和 `action_sent_s`。
3. 写 dataset 时，查找 `action_sent_s <= state_sample_s` 的最近一条 action。
4. 图像、这个 state 和查到的 action 一起组成数据帧。

每次录制还会生成：

```text
logs/gello_record_sync_<时间>.csv
```

重点看这些列：

- `action_age_ms`：state 采样时，这条 action 已经发送多久。它应该非负，通常小于一个 60 Hz 周期。
- `state_to_observation_end_ms`：state 采样后，两路相机读取和整理用了多久。
- `camera_timings`：每路相机调用的起止时间。
- `frame_loop_ms`：这一帧写入前的主循环工作时间。

实际日志验证过，正常帧的 `action_age_ms` 通常约 1–10 ms，没有选择未来 action。

相机硬件配置是 30 Hz，所以 dataset 也必须写成 30 Hz。之前 dataset 标成 60 Hz、实际只能得到约 30 帧，会让数据时间轴看起来快一倍。现在控制频率和录制频率已经分开：

```text
机械臂控制：60 Hz
图像/state/action 采样：30 Hz
```

30 Hz 数据集每帧记录的是该采样时刻有效的 60 Hz 控制命令，这是正常的降采样，不是不同步。

## 四、夹爪 Error 19 是怎么消除的

现象是只要连续控制夹爪，就出现：

```text
set_rs485_data -> code=1
controller_error=19
```

排查过程里先试过降低夹爪命令频率：2 Hz、5 Hz、6.7 Hz 都不报错，但低频带来明显跟手延迟。继续对照实验后发现，真正与故障一致的变量不是频率，而是夹爪速度：

- 报错时使用默认最大速度 5000。
- 速度降到 1500 后，从 2 Hz 一直提高到 20 Hz 都稳定。
- 最后恢复到最高 60 Hz，仍然稳定。

运行期发送也改成 SDK 专用的非阻塞 `set_gripper_position`，不再手工拼通用 RS485 数据包；持续 ServoJ 时关闭 `wait_motion`，否则 SDK 会等待机械臂停止。

最终参数为：

```yaml
gripper_speed: 1500
gripper_command_interval_s: 0.0166667
gripper_command_threshold: 0.01
```

这里的 60 Hz 是“最多检查和发送 60 次”。当夹爪目标变化不足 `0.01` 时会去重，不会发送没有意义的重复 RS485 命令。

每次成功或失败的夹爪命令都会写入：

```text
logs/xarm7_gripper_errors.log
```

成功记录包含目标值、脉冲位置、调用耗时和返回码。最终实验中单次调用通常只需要约 1.3–2.2 ms，60 Hz、速度 1500 时没有再次出现 C19。

如果以后更换夹爪、线缆或固件后 C19 再次出现，应先把速度降下来验证。如果低速也报错，就应该检查腕部线缆、接头、末端供电和末端 IO 板固件，而不是无限降低控制频率。

## 五、运行和验收

正式录制：

```bash
uv run record \
  --config_path config/gello/xarm7_gello_record_config.yaml
```

每次修改控制或相机配置后，至少检查：

1. 遥操作全过程没有肉眼可见的停顿或追赶。
2. J7 可以独立旋转。
3. 接近 TCP 高度下限时本地投影生效，控制器 Safety Boundary 保持开启。
4. 同步日志中 `action_age_ms` 不为负，绝大多数小于 16.67 ms。
5. 实际 dataset 是约 30 FPS，元数据也为 30 FPS。
6. 连续开合夹爪时没有新增 Error 19。

不要提交运行生成的 CSV 或设备错误日志；它们用于本机诊断，不属于训练数据和源代码。
