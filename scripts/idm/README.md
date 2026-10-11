# IDM 动作回放

`replay_IDM_output.py` 使用 xArm mode 7 回放 TCP 目标，默认下发频率为 25 Hz，可通过 `--fps` 修改。

从项目根目录运行：

```bash
# 校验 IDM 的 10 维输出，不连接机械臂
.venv/bin/python scripts/idm/replay_IDM_output.py /path/to/output.npy \
  --format tcp-rot6d-m --fps 15 --dry-run

# 回放相同文件
.venv/bin/python scripts/idm/replay_IDM_output.py /path/to/output.npy \
  --format tcp-rot6d-m --fps 15

# 列出指定目录中的 *_tcp_action.npy 动作文件
.venv/bin/python scripts/idm/replay_IDM_output.py --dir /path/to/actions --list
```

支持两种输入格式：

| `--format` | 每帧数据 |
| --- | --- |
| `tcp-rpy-mm`（默认） | 7 维：位置（mm）、RPY（rad）、夹爪开口（mm） |
| `tcp-rot6d-m` | 10 维：位置（m）、旋转矩阵前两列 `[r11,r21,r31,r12,r22,r32]`、夹爪闭合比例（0 全开，1 全闭） |

位置均为机械臂基坐标系下的绝对 TCP 目标。脚本将旋转转换为轴角，并将夹爪闭合比例裁剪到 `[0,1]`、转换为 G2 的 0–84 mm 开口。

可直接传入 `.npy` 或 `.csv` 文件路径（CSV 第一行为表头），也可传入动作名称。不指定 `--dir` 时，依次查找 `scripts/idm/`、`scripts/`、项目根目录下的 `action1/` 和 `真机可跑action/` 中的 `*_tcp_action.npy` 文件。
