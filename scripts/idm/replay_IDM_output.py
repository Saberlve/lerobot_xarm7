#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_IDM_output.py — 用 xArm 笛卡尔在线轨迹规划（set_mode(7) + set_position_aa）
以指定频率回放 TCP 动作 chunk（默认 25Hz）。

数据格式: (N, 7) = [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad, gripper_mm]
  - x/y/z   : 基坐标系 TCP 位置, mm
  - rx/ry/rz: xArm 控制器上报的 RPY (roll/pitch/yaw), rad。
              注意: set_position_aa 要的是轴角, 脚本内部做 RPY -> 轴角转换
              (数据里 roll 贴着 ±pi, 通用轴角提取公式在 pi 附近奇异, 已特判)
  - gripper : xArm Gripper G2 开口, mm (0-84)

--format tcp-rot6d-m 支持 (N, 10):
  [x_m, y_m, z_m, r11, r21, r31, r12, r22, r32, gripper_norm]
  位置为基坐标系绝对目标；夹爪 0=全开、1=全闭，超出 [0,1] 的值裁剪。

用法:
  python scripts/idm/replay_IDM_output.py sgrasp_ep00000_w0000
  python scripts/idm/replay_IDM_output.py /home/wsx/下载/ema05.npy --format tcp-rot6d-m --fps 15 --dry-run
  python scripts/idm/replay_IDM_output.py --list

运行环境 (xarm SDK + numpy):
  /usr/share/EmbodiedAI/VLArmory/examples/realRobots/xArm7/lerobot_xarm7/.venv/bin/python
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
# 优先查找 IDM 目录，同时保留移动前的数据目录；也可用 --dir 显式指定。
DATA_DIR_CANDIDATES = [
    SCRIPT_DIR,
    SCRIPTS_DIR,
    PROJECT_ROOT / "action1",
    PROJECT_ROOT / "真机可跑action",
]

FPS = 25.0                    # 下发频率 Hz
PERIOD = 1.0 / FPS
APPROACH_SPEED = 50.0         # 对齐第一个目标点时的速度 mm/s (mode 0 阻塞运动)
STREAM_SPEED = 200.0          # 在线规划下发的参考速度 mm/s
STREAM_ACC = 2000.0           # mm/s^2
GRIPPER_SPEED = 100           # G2 夹爪速度 mm/s (15-225)
GRIPPER_FORCE = 50            # G2 夹爪力 1-100
GRIPPER_MIN_MM = 0.0
GRIPPER_MAX_MM = 84.0
GRIPPER_RESEND_THRESH_MM = 0.3   # 变化超过该值才重新下发 (RS485 带宽有限)
GRIPPER_MIN_INTERVAL_S = 0.1     # 夹爪指令最小间隔
MAX_STEP_JUMP_MM = 40.0          # 相邻目标点距离超过该值时告警
MIN_TCP_Z_MM = -5.0              # TCP 高度下限 (基坐标系 mm), 低于则拒绝执行


# ---------------------------------------------------------------- 数据加载

def resolve_data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    for d in DATA_DIR_CANDIDATES:
        if d.is_dir() and list(d.glob("*_tcp_action.npy")):
            return d
    return SCRIPT_DIR


def list_chunks(data_dir: Path) -> list[str]:
    names = sorted(p.name[: -len("_tcp_action.npy")] for p in data_dir.glob("*_tcp_action.npy"))
    return names


def load_chunk(name: str, data_dir: Path, data_format: str = "tcp-rpy-mm") -> np.ndarray:
    """按名字加载 chunk, 接受完整文件名或裸 sample_id。"""
    base = name
    for suffix in ("_tcp_action.npy", "_tcp_action.csv", ".npy", ".csv"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.endswith("_tcp_action"):
        base = base[: -len("_tcp_action")]

    explicit = Path(name).expanduser()
    if not explicit.is_file():
        explicit = data_dir / explicit
    npy = explicit if explicit.is_file() and explicit.suffix == ".npy" else data_dir / f"{base}_tcp_action.npy"
    csv_path = explicit if explicit.is_file() and explicit.suffix == ".csv" else data_dir / f"{base}_tcp_action.csv"
    if npy.exists():
        data = np.load(npy, allow_pickle=False)
    elif csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        data = np.asarray([[float(v) for v in row] for row in rows[1:]], dtype=np.float64)
    else:
        raise FileNotFoundError(f"找不到 chunk: {name} (尝试过 {npy.name} / {csv_path.name})")

    columns = 10 if data_format == "tcp-rot6d-m" else 7
    if data.ndim != 2 or data.shape[1] != columns or len(data) == 0:
        raise ValueError(f"数据应为非空 (N, {columns}), 实际 {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("数据含 NaN/Inf")
    return data


def convert_rot6d_chunk(data: np.ndarray) -> np.ndarray:
    """Metres + column-based 6D rotation + closure -> mm + axis-angle + G2 opening."""
    converted = np.empty((len(data), 7), dtype=np.float64)
    converted[:, :3] = data[:, :3] * 1000.0
    converted[:, 6] = GRIPPER_MAX_MM * (1.0 - np.clip(data[:, 9], 0.0, 1.0))
    previous = None
    for i, row in enumerate(data):
        a1, a2 = np.asarray(row[3:9], dtype=np.float64).reshape(2, 3)
        norm1 = np.linalg.norm(a1)
        if norm1 < 1e-8:
            raise ValueError(f"step {i}: 6D 旋转第一列为零")
        b1 = a1 / norm1
        b2 = a2 - np.dot(b1, a2) * b1
        norm2 = np.linalg.norm(b2)
        if norm2 < 1e-8:
            raise ValueError(f"step {i}: 6D 旋转两列共线")
        b2 /= norm2
        rotation = np.column_stack((b1, b2, np.cross(b1, b2)))
        aa = matrix_to_axis_angle(rotation)
        angle = np.linalg.norm(aa)
        if previous is not None and angle > 1e-9:
            alternative = -aa / angle * (2.0 * math.pi - angle)
            if np.linalg.norm(alternative - previous) < np.linalg.norm(aa - previous):
                aa = alternative
        converted[i, 3:6] = aa
        previous = aa
    return converted


# ------------------------------------------------------- RPY -> 轴角转换

def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """xArm RPY 约定: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。
    (与 lerobot_xarm7 local_kinematics.py 的 _rpy_rotation 一致)"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    cos_angle = min(1.0, max(-1.0, (np.trace(R) - 1.0) / 2.0))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle > 1e-4:
        axis = np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        ) / (2.0 * math.sin(angle))
        return axis * angle
    # angle ≈ pi: sin(angle)≈0, 上式奇异。此时 R + I = 2 n n^T, 从对角元恢复转轴。
    M = (R + np.eye(3)) / 2.0
    i = int(np.argmax(np.diag(M)))
    ni = math.sqrt(max(M[i, i], 0.0))
    if ni < 1e-9:
        return np.zeros(3)  # 理论上不会发生
    n = np.zeros(3)
    n[i] = ni
    for j in range(3):
        if j != i:
            n[j] = M[i, j] / ni  # M 对称, n_j = M_ij / n_i (整体符号任意, pi 旋转等价)
    return n * angle


def pose_rpy_to_aa(pose6: np.ndarray) -> list[float]:
    """[x,y,z,roll,pitch,yaw](mm, rad) -> set_position_aa 的轴角位姿。"""
    aa = matrix_to_axis_angle(rpy_to_matrix(*pose6[3:6]))
    return [float(pose6[0]), float(pose6[1]), float(pose6[2]),
            float(aa[0]), float(aa[1]), float(aa[2])]


# ---------------------------------------------------------------- 校验

def sanity_check(data: np.ndarray) -> list[str]:
    warns = []
    z_min = float(data[:, 2].min())
    if z_min < MIN_TCP_Z_MM:
        warns.append(f"TCP z 最低 {z_min:.1f} mm, 低于下限 {MIN_TCP_Z_MM} mm")
    steps = np.linalg.norm(np.diff(data[:, :3], axis=0), axis=1)
    jump = float(steps.max()) if steps.size else 0.0
    if jump > MAX_STEP_JUMP_MM:
        warns.append(f"相邻点最大步进 {jump:.1f} mm (> {MAX_STEP_JUMP_MM} mm)")
    g = data[:, 6]
    if g.min() < GRIPPER_MIN_MM - 1e-6 or g.max() > GRIPPER_MAX_MM + 1e-6:
        warns.append(f"夹爪范围 [{g.min():.1f}, {g.max():.1f}] mm 超出 G2 量程 [0, 84]")
    return warns


# ---------------------------------------------------------------- 主流程

def main() -> int:
    parser = argparse.ArgumentParser(description="回放 TCP 动作 chunk (xArm mode 7 在线轨迹规划)")
    parser.add_argument("chunk", nargs="?", help="chunk 名或 .npy/.csv 文件路径")
    parser.add_argument("--format", choices=("tcp-rpy-mm", "tcp-rot6d-m"), default="tcp-rpy-mm", help="7维 mm/RPY/夹爪mm，或10维 m/6D旋转/夹爪闭合比例")
    parser.add_argument("--list", action="store_true", help="列出所有可用 chunk")
    parser.add_argument("--ip", default="192.168.1.245", help="xArm 控制器 IP")
    parser.add_argument("--dir", default=None, help="数据目录 (默认自动探测 scripts/idm、scripts 和项目根目录下的 action1 / 真机可跑action)")
    parser.add_argument("--fps", type=float, default=FPS, help=f"下发频率 (默认 {FPS:g}Hz)")
    parser.add_argument("--dry-run", action="store_true", help="只加载校验和打印, 不连接机械臂")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps 必须为正有限数")

    data_dir = resolve_data_dir(args.dir)
    if args.list:
        print(f"# 数据目录: {data_dir}")
        for n in list_chunks(data_dir):
            print(n)
        return 0
    if not args.chunk:
        parser.error("请指定 chunk 名, 或用 --list 查看")

    data = load_chunk(args.chunk, data_dir, args.format)
    if args.format == "tcp-rot6d-m":
        clipped = int(np.count_nonzero((data[:, 9] < 0) | (data[:, 9] > 1)))
        print(f"[format] m/6D旋转/夹爪闭合比例 -> mm/轴角/G2开口; {clipped} 帧夹爪值裁剪到 [0,1]")
        data = convert_rot6d_chunk(data)
    n = len(data)
    period = 1.0 / args.fps
    print(f"[load] {args.chunk}: {n} 步, {n / args.fps:.2f} s @ {args.fps:g} Hz")
    print(f"[load] 起点 TCP: {np.round(data[0, :3], 1)}  终点 TCP: {np.round(data[-1, :3], 1)}")
    print(f"[load] 夹爪范围: {data[:, 6].min():.1f} ~ {data[:, 6].max():.1f} mm")

    warns = sanity_check(data)
    for w in warns:
        print(f"[warn] {w}")
    if any("TCP z" in w for w in warns):
        print("[abort] 存在低于安全高度的目标点, 拒绝执行。")
        return 2

    # 预转换全部轴角位姿
    aa_targets = (data[:, :6].tolist() if args.format == "tcp-rot6d-m"
                  else [pose_rpy_to_aa(row[:6]) for row in data])

    if args.dry_run:
        print("[dry-run] 前 3 步轴角位姿:")
        for i, aa in enumerate(aa_targets[:3]):
            print(f"  step {i}: {[round(v, 4) for v in aa]}")
        return 0

    from xarm.wrapper import XArmAPI

    arm = XArmAPI(args.ip)
    try:
        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)
        if arm.error_code != 0:
            raise RuntimeError(f"控制器存在错误码 {arm.error_code}, 请先处理")

        # 读取当前 TCP, 报告与起点的距离
        code, cur = arm.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_position 失败, code={code}")
        dist = float(np.linalg.norm(np.asarray(cur[:3]) - data[0, :3]))
        print(f"[init] 当前 TCP {np.round(cur[:3], 1)}, 距起点 {dist:.1f} mm")

        # 夹爪初始化 (G2), 先到 chunk 起始开口
        g0 = float(np.clip(data[0, 6], GRIPPER_MIN_MM, GRIPPER_MAX_MM))
        arm.set_gripper_enable(True)
        arm.set_gripper_mode(0)
        arm.set_gripper_g2_position(g0, speed=GRIPPER_SPEED, force=GRIPPER_FORCE, wait=True, timeout=10)

        # mode 0 慢速对齐到第一个目标点
        print(f"[init] 以 {APPROACH_SPEED:g} mm/s 对齐到起点...")
        code = arm.set_position_aa(aa_targets[0], speed=APPROACH_SPEED, mvacc=STREAM_ACC,
                                   is_radian=True, wait=True, timeout=60)
        if code != 0:
            raise RuntimeError(f"对齐起点失败, code={code}")

        # 进入笛卡尔在线轨迹规划模式 (mode 7)
        code = arm.set_mode(7)
        if code != 0:
            raise RuntimeError(f"set_mode(7) 失败, code={code}")
        code = arm.set_state(0)
        if code != 0:
            raise RuntimeError(f"set_state(0) 失败, code={code}")
        time.sleep(0.2)

        print(f"[run] 开始 {args.fps:g} Hz 下发, 共 {n} 步 (Ctrl+C 急停)")
        last_gripper = g0
        last_gripper_t = 0.0
        t0 = time.perf_counter()
        for i, aa in enumerate(aa_targets):
            code = arm.set_position_aa(aa, speed=STREAM_SPEED, mvacc=STREAM_ACC,
                                       is_radian=True, wait=False)
            if code != 0:
                raise RuntimeError(f"step {i} set_position_aa 失败, code={code}")
            if arm.error_code != 0:
                raise RuntimeError(f"step {i} 控制器报错, error_code={arm.error_code}")

            # 夹爪限频下发
            g = float(np.clip(data[i, 6], GRIPPER_MIN_MM, GRIPPER_MAX_MM))
            now = time.perf_counter()
            if (abs(g - last_gripper) >= GRIPPER_RESEND_THRESH_MM
                    and now - last_gripper_t >= GRIPPER_MIN_INTERVAL_S):
                arm.set_gripper_g2_position(g, speed=GRIPPER_SPEED, force=GRIPPER_FORCE, wait=False)
                last_gripper = g
                last_gripper_t = now

            # 按指定频率下发
            next_t = t0 + (i + 1) * period
            remain = next_t - time.perf_counter()
            if remain > 0:
                time.sleep(remain)

        elapsed = time.perf_counter() - t0
        print(f"[done] {n} 步完成, 实际用时 {elapsed:.2f} s (理论 {n * period:.2f} s, 均频 {n / elapsed:.1f} Hz)")
        arm.set_mode(0)
        arm.set_state(0)
        return 0

    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C, 急停")
        arm.set_state(4)
        arm.set_mode(0)
        return 130
    except Exception as e:
        print(f"[error] {e}")
        arm.set_state(4)
        arm.set_mode(0)
        return 1
    finally:
        arm.disconnect()


if __name__ == "__main__":
    sys.exit(main())
