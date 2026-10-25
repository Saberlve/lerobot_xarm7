# Eval 配置

按 `模型/任务/模型_任务.yaml` 保存，模型为 `ACT`、`pi05`、`N0`、`FTP1`。
当前每个模型有 8 个任务：`egg_pick`、`chip_pick`、`charger_insert`、
`whiteboard_erase`、`pullback_car`、`nut_removal`、`nut_tighten`、`charger_press`。
保留 `charger_insert` 和 `charger_press`，不包含 `handwash_press`。

通常只需关注配置顶部两个参数：

- `action_chunk_size`：与模型训练的 `chunk_size` / `action_horizon` 一致，用于检查服务端模型是否匹配；修改此值不会改变模型输出长度。
- `steps_per_inference`：四种模型均按任务设置：chip_pick=23、egg_pick=13、charger_press=10、pullback_car=13、whiteboard_erase=25、nut_removal=13、nut_tighten=13、charger_insert=23；表示执行相应数量的动作步后重新推理，并非模型内部的去噪迭代次数。所有任务关闭 temporal ensemble。

下表为各模型的 `action_chunk_size`（训练输出长度）：

| 任务 | ACT | pi05 | N0 | FTP1 |
|---|---|---|---|---|
| egg_pick | 15 | 15 | 15 | 15 |
| chip_pick | 25 | 25 | 25 | 25 |
| charger_insert | 25 | 25 | 25 | 25 |
| whiteboard_erase | 30 | 30 | 30 | 30 |
| pullback_car | 15 | 15 | 15 | 15 |
| nut_removal | 15 | 15 | 15 | 15 |
| nut_tighten | 15 | 15 | 15 | 15 |
| charger_press | 10 | 10 | 10 | 10 |

每份 YAML 首行标注对应的专用任务训练配置，长度均取自该文件。
语言指令来自对应的 GELLO 任务配置。新配置沿用已有硬件设置与 `dry_run: true`。
根目录原有 YAML 保留供旧命令和测试使用。

普通 EEF 评测默认关闭动作曲线记录；设置 `enable_logs: true` 开启（`log_dir: logs`）。
每次运行创建独立的 `logs/eef_eval_<时间>_<随机后缀>/`，相对路径以启动目录为准；
开启后终端会打印绝对路径。可用 `enable_logs: false` 关闭。

- `actions.csv`：每个执行步记录 episode、step、时间戳、10 维模型动作、下发前实际反馈、
  关节角、控制命令和 SDK dispatch 信息。XYZ 统一为 mm，关节角为 rad，旋转为列向量 6D。
  `elapsed_s` 是运行内相对单调时间，反馈使用缓存采样时间。反馈不可用时留空并记录 `feedback_error`。
- `episode_001_curves.svg` 等：退出或结束时按 episode 生成 10 维动作与反馈曲线，可直接用浏览器打开。
  CSV 逐行写入，强制杀进程时仍可读取已写入的数据，但可能没有 SVG。
- temporal ensemble 模式记录融合后的控制动作，chunk 编号、offset 和单次推理耗时留空。
  `dry_run` 同样记录，但不会下发控制；控制命令和 SDK 接受状态不代表目标已到达。

记录读取现有实时状态缓存，不增加相机采集或控制器 RPC。夹爪专项保留原有独立日志。

在 lerobot_xarm7 目录运行，例如：

```bash
.venv/bin/python scripts/uf_starvla_eval.py --config_path config/eval/pi05/charger_press/pi05_charger_press.yaml --check_config=true
```
