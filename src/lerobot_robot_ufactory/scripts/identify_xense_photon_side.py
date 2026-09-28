"""Identify which Xense Photon serial number is mounted on the left.

Run on the computer connected to both sensors. This script reads images only.
"""

import importlib.util
import os
from pathlib import Path
import sys
import time


def use_project_python_if_needed():
    missing = [name for name in ("numpy", "xensesdk") if importlib.util.find_spec(name) is None]
    if not missing:
        return
    project_python = Path(__file__).resolve().parents[3] / ".venv/bin/python"
    if project_python.is_file() and os.path.abspath(sys.executable) != str(project_python):
        print(f"当前 Python 缺少 {', '.join(missing)}，切换到 {project_python}", flush=True)
        os.execv(str(project_python), [str(project_python), __file__, *sys.argv[1:]])
    raise SystemExit(f"当前 Python 缺少 {', '.join(missing)}；请安装项目的 xense 依赖。")


use_project_python_if_needed()

import numpy as np
from xensesdk import Sensor


SERIALS = ("OG001931", "OG001932")


def read_frame(sensor):
    frame, _ = sensor.selectSensorInfo(
        sensor.OutputType.Rectify, sensor.OutputType.TimeStamp
    )
    if frame is None:
        raise RuntimeError("SDK returned no Rectify image")
    return np.asarray(frame, dtype=np.int16)


def main():
    found = Sensor.scanSerialNumber()
    print(f"检测到的 Photon 传感器：{found}")
    missing = [serial for serial in SERIALS if serial not in found]
    if missing:
        raise RuntimeError(f"未检测到传感器：{', '.join(missing)}")

    sensors = {}
    try:
        for serial in SERIALS:
            sensor = Sensor.create(serial, disable_infer=True, infer_mode="fast")
            if sensor is None:
                raise RuntimeError(f"无法打开 {serial}")
            sensors[serial] = sensor

        print("请保持两只传感器不受触碰，2 秒后采集静止画面。")
        time.sleep(2)
        baseline = {serial: read_frame(sensor).copy() for serial, sensor in sensors.items()}

        input("准备好后按回车，然后在 10 秒内反复按压、松开【物理左侧】传感器……")
        peaks = {serial: 0.0 for serial in SERIALS}
        end = time.monotonic() + 10
        while time.monotonic() < end:
            scores = {}
            for serial, sensor in sensors.items():
                frame = read_frame(sensor)
                if frame.shape != baseline[serial].shape:
                    raise RuntimeError(f"{serial} 的图像尺寸发生变化")
                # Mean absolute pixel change from the untouched image.
                scores[serial] = float(np.abs(frame - baseline[serial]).mean())
                peaks[serial] = max(peaks[serial], scores[serial])
            print("  ".join(f"{serial}: {scores[serial]:6.2f}" for serial in SERIALS), flush=True)
            time.sleep(0.2)

        ranked = sorted(SERIALS, key=lambda serial: peaks[serial], reverse=True)
        first, second = ranked
        print("\n最大画面变化量：")
        for serial in SERIALS:
            print(f"  {serial}: {peaks[serial]:.2f}")
        if peaks[first] >= 2 * max(peaks[second], 0.1) and peaks[first] >= 1.0:
            print(f"左侧传感器的序列号很可能是：{first}")
        else:
            print("结果不明确。请观察按压左侧时哪个数值上升，再试一次。")
    finally:
        for sensor in sensors.values():
            sensor.release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已停止。")
