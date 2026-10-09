# GELLO 任务配置

每个任务一个子文件夹。配置是当前 `../xarm7_gello_base.yaml` 的独立副本；后续修改 base 不会自动同步。

共享设置：控制 30 Hz，RGB/数据集 15 Hz，Photon 60 Hz；记录关节和 TCP（both）；开启 GELLO 七轴电流控制，沿用当前 base 的 `experimental: false`；夹爪由键盘控制，夹爪监测关闭；结束录制后计算并保存 Mesh3DFlow。

| 任务 | 配置 | 数据集保存路径 | repo_id |
|---|---|---|---|
| 鸡蛋抓取 | [xarm7_gello_egg_pick.yaml](egg_pick/xarm7_gello_egg_pick.yaml) | `datasets/xarm7_gello_tasks/egg_pick` | `ufactory/xarm7_gello_egg_pick` |
| chip 抓取 | [xarm7_gello_chip_pick.yaml](chip_pick/xarm7_gello_chip_pick.yaml) | `datasets/xarm7_gello_tasks/chip_pick` | `ufactory/xarm7_gello_chip_pick` |
| 按压洗手液 | [xarm7_gello_handwash_press.yaml](handwash_press/xarm7_gello_handwash_press.yaml) | `datasets/xarm7_gello_tasks/handwash_press` | `ufactory/xarm7_gello_handwash_press` |
| 齿轮装配 | [xarm7_gello_gear_assembly.yaml](gear_assembly/xarm7_gello_gear_assembly.yaml) | `datasets/xarm7_gello_tasks/gear_assembly` | `ufactory/xarm7_gello_gear_assembly` |
| 擦白板 | [xarm7_gello_whiteboard_erase.yaml](whiteboard_erase/xarm7_gello_whiteboard_erase.yaml) | `datasets/xarm7_gello_tasks/whiteboard_erase` | `ufactory/xarm7_gello_whiteboard_erase` |
| 弹簧小车 | [xarm7_gello_spring_cart.yaml](spring_cart/xarm7_gello_spring_cart.yaml) | `datasets/xarm7_gello_tasks/spring_cart` | `ufactory/xarm7_gello_spring_cart` |
| 拧松螺母 | [xarm7_gello_nut_loosen.yaml](nut_loosen/xarm7_gello_nut_loosen.yaml) | `datasets/xarm7_gello_tasks/nut_loosen` | `ufactory/xarm7_gello_nut_loosen` |
| 大号 USB 插入 | [xarm7_gello_usb_insert.yaml](usb_insert/xarm7_gello_usb_insert.yaml) | `datasets/xarm7_gello_tasks/usb_insert` | `ufactory/xarm7_gello_usb_insert` |

从项目根目录启动，例如：

```bash
.venv/bin/uf-lerobot-record --config_path config/gello/tasks/egg_pick/xarm7_gello_egg_pick.yaml
```

保存路径和电流 profile 路径均相对于启动目录。

弹簧小车只提供了任务名称，目前使用通用描述 `Manipulate the spring-loaded cart.`，具体运动方向和终止条件需补充。USB 插入配置以完全插入并短暂保持为结束条件。

所有任务沿用 base 的 `min_tcp_z_mm: 2`，实际接触高度需按各任务布置确认。鸡蛋配置同时适用于生鸡蛋和熟鸡蛋；如需分别统计，可复制配置并为两组设置不同的数据集路径和 repo_id。
