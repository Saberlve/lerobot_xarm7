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

在 lerobot_xarm7 目录运行，例如：

```bash
.venv/bin/python scripts/uf_starvla_eval.py --config_path config/eval/pi05/charger_press/pi05_charger_press.yaml --check_config=true
```
