"""Parse the real EEF robot configuration without constructing any devices."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import draccus
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from eef_executor import decode_robot_config, make_robot


class EEFRobotConfigTests(unittest.TestCase):
    def encoded(self):
        cfg = UFRobotConfig(robot_dof=7, record_space='both', min_tcp_z_mm=-2,
            cameras={name: RealSenseCameraConfig(serial_number_or_name=serial, width=640, height=480, fps=15)
                     for name, serial in [('camera', '242622070583'), ('camera2', '148522072685')]})
        raw = draccus.encode(cfg)
        raw['type'] = cfg.type
        for name, camera in cfg.cameras.items():raw['cameras'][name]['type'] = camera.type
        return raw

    def test_encoded_defaults_decode_with_null_camera_args(self):
        raw = self.encoded()
        self.assertIsNone(raw['cameras_args'])
        decoded = decode_robot_config(SimpleNamespace(robot=raw, dry_run=True))
        self.assertIsNone(decoded.cameras_args)
        self.assertEqual(set(decoded.cameras), {'camera', 'camera2'})
        self.assertEqual(decoded.cameras['camera2'].serial_number_or_name, '148522072685')
        self.assertTrue(decoded.no_action)
        self.assertFalse(raw['no_action'])

    def test_missing_and_dictionary_camera_args_remain_supported(self):
        for value in [None, {}, {'custom': 1}]:
            with self.subTest(value=value):
                raw = self.encoded()
                if value is None:raw.pop('cameras_args')
                else:raw['cameras_args'] = value
                decoded = decode_robot_config(SimpleNamespace(robot=raw, dry_run=False))
                self.assertEqual(decoded.cameras_args, value)
                self.assertFalse(decoded.no_action)

    def test_unified_joint_entry_preserves_typed_robot_decoding(self):
        from copy import deepcopy
        from lerobot.robots import RobotConfig
        from uf_starvla_eval import StarVLAEvalConfig, _joint_config
        raw = self.encoded()
        before = deepcopy(raw)
        expected = draccus.decode(RobotConfig, raw)
        cfg = StarVLAEvalConfig(robot=raw, action_space="joint", dry_run=True)
        result = _joint_config(cfg)
        self.assertEqual(draccus.encode(result.robot), draccus.encode(expected))
        self.assertEqual(raw, before)
        self.assertFalse(result.robot.no_action)

    def test_robot_factory_receives_decoded_config_without_connecting(self):
        cfg = SimpleNamespace(robot=self.encoded(), dry_run=True)
        with patch('lerobot_robot_ufactory.robots.uf_robot.uf_robot.UFRobot') as robot:
            result = make_robot(cfg)
            self.assertIs(result, robot.return_value)
            self.assertIsInstance(robot.call_args.args[0], UFRobotConfig)
            robot.return_value.connect.assert_not_called()


if __name__ == '__main__':unittest.main()
