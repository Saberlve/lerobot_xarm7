# GELLO 重力补偿代码结构

```text
gravity_compensation/
├── config.py                    # 设备配置、公共参数校验
├── control/                     # 出力计算与控制生命周期
│   ├── model.py                 # 重力、阻尼和电流变化率
│   ├── runtime.py               # 控制循环、保护、遥操适配
│   └── tuning.py                # 调参会话、只读与补偿交接
├── hardware/
│   └── transport.py             # Dynamixel 通信与寄存器操作
├── monitoring/
│   ├── encoder_monitor.py       # 只读角度采样
│   └── logging.py               # 独立线程写诊断日志
├── models/
│   ├── geometry.py              # 网页使用的 URDF 几何解析
│   ├── mesh_mass.py             # STL 质量、质心与惯量
│   └── urdf.py                  # 简化 URDF 生成
├── official_model.py            # 离线模型生成工具，不连接硬件
└── web/
    ├── tuning_web.py            # HTTP 接口、统一网页入口
    ├── model_web.py             # 浏览器模型导出、HTML 组装
    └── viewer_assets/           # HTML、JS、Three.js 和许可证
```

补偿模块直接位于本包下，不再包裹 `core` 层。
`web` 和 GELLO 遥操作调用 `control`、`hardware`、`monitoring` 与 `models`；
这些模块不依赖网页服务或前端资源。导入子包不会连接硬件。

重力补偿只保留网页和遥操作接入（包括通过 GELLO 遥操作录制）。
独立诊断/出力 CLI 已移除。只读角度、模型姿态检查和持续补偿统一使用网页。
诊断日志保留在 `monitoring/logging.py`，磁盘写入在独立线程进行；
写入失败会请求停止补偿。

## 网页入口

```bash
.venv/bin/python -m lerobot_robot_ufactory.gravity_compensation.web.tuning_web \
  --profile config/gravity/gello_A_working.yaml --port 8765
```

安装后的网页启动命令为 `uf-gello-tune-web`。
默认离线查看，不连接串口；点击只读或补偿按钮才接入 GELLO。
增益、变化率、温度和卸力操作说明见网页内提示。

## 遥操作接入

```bash
.venv/bin/uf-robot-teleop \
  --config_path config/gello/xarm7_gello_teleop_gravity.yaml
```

通过 `teleop.gravity_compensation` 配置接入，补偿线程独占串口，
遥操作使用缓存角度。退出时卸力。录制使用
`config/gello/xarm7_gello_record_gravity_config.yaml` 中的同一接入路径。

## 离线模型生成

`official_model.py` 仅用于重建 STL 装配模型，不是重力补偿运行入口。
现有网页与遥操直接加载配置指定的 URDF。需要重建时，从项目根目录运行
（输出目录必须是新目录）：

```bash
.venv/bin/python -m lerobot_robot_ufactory.gravity_compensation.official_model \
  --spec config/gravity/model/estimate.json \
  --output-dir /tmp/gello-model-regenerated
```
