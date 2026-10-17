"""Hardware-free regressions for evaluation and replay control flow.

Load the actual functions via AST so these checks need neither robot SDKs nor
model weights. Only external dependencies are replaced with fakes.
"""
import ast
import __future__
import logging
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'src/lerobot_robot_ufactory/scripts/uf_lerobot_eval.py'
REPLAY = ROOT / 'src/lerobot_robot_ufactory/scripts/uf_lerobot_replay.py'
STARVLA = ROOT / 'scripts/uf_starvla_eval.py'


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == name]
    exec(compile(tree, str(path), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


class Clock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 0.0)


class ControlRegressions(unittest.TestCase):
    def action_conversion(self, kind, relative, continuous=False):
        tree = ast.parse(EVAL.read_text())
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and any(isinstance(child, ast.Name) and child.id == 'rot_delta'
                            for child in ast.walk(node)))
        action = dict(zip(['pose.x', 'pose.y', 'pose.z', 'pose.rx', 'pose.ry', 'pose.rz'],
                          [2., 3., 4., 0.1, 0.2, 0.3]))
        if continuous:
            action['pose.rx'] = 3.5
        namespace = dict(keys=[''], curr_robot_dict={'': {'type': kind}},
                         prev_action_dict={'': {'type': kind, 'pose': [300, 200, 100, 0, 0, 0]}},
                         robot_action_to_send=action, relative=relative, rx_continuous=continuous,
                         np=NS(array=list, pi=__import__('math').pi), math=__import__('math'),
                         compute_target_axis_angle=lambda prev, delta: [a+b for a,b in zip(prev,delta)])
        exec(compile(ast.Module(body=[loop], type_ignores=[]), str(EVAL), 'exec'), namespace)
        return action

    def test_relative_cartesian_action_is_reconstructed(self):
        action = self.action_conversion(1, True)
        self.assertEqual([action['pose.'+k] for k in ('x','y','z')], [302.,203.,104.])
        self.assertEqual(action['pose.rx'], 0.1)

    def test_joint_action_skips_cartesian_conversion(self):
        self.assertEqual(self.action_conversion(0, True)['pose.x'], 2.)

    def test_continuous_rotation_is_restored(self):
        self.assertAlmostEqual(self.action_conversion(1, False, True)['pose.rx'], 3.5-2*__import__('math').pi)

    def run_replay(self, delays):
        clock = Clock()
        sent = []
        def send(action):
            sent.append((clock.now, action))
            clock.now += delays.get(len(sent), 0)
        ns = dict(time=clock, precise_sleep=clock.sleep, state_to_robot_action=lambda value:value)
        replay = load_function(REPLAY, 'replay_episode', ns)
        replay(NS(send_action=send), NS(fps=30, states=list(range(40))), show_progress=False)
        return sent

    def test_replay_first_move_does_not_catch_up(self):
        sent = self.run_replay({1: 1.0})
        self.assertGreaterEqual(sent[1][0], 1.0+1/30-1e-9)
        self.assertTrue(all(b[0]-a[0] >= 1/30-1e-9 for a,b in zip(sent,sent[1:])))
        self.assertEqual([a for _,a in sent], list(range(40)))

    def test_replay_mid_episode_overrun_does_not_catch_up(self):
        sent = self.run_replay({10: 0.5})
        self.assertTrue(all(b[0]-a[0] >= 1/30-1e-9 for a,b in zip(sent,sent[1:])))

    def test_replay_preserves_normal_fps(self):
        sent = self.run_replay({})
        self.assertAlmostEqual(sent[-1][0], 39/30)

    def run_eval(self, error=None, disconnect_error=None, normal_exit=False, connect_error=None):
        robot = Mock()
        robot.get_observation.return_value = {}
        robot.reset_to_initial.side_effect = error
        robot.disconnect.side_effect = disconnect_error
        robot.connect.side_effect = connect_error
        listener = Mock()
        policy = Mock(config=NS(device='cpu'))
        processor = Mock()
        metadata = NS(features={}, fps=30, stats={})
        cfg = NS(robot=NS(), dataset=NS(repo_id='test',root=None,fps=30,rename_map={}),
                 policy=NS(pretrained_path=None))
        ns = dict(init_logging=lambda:None, logging=logging, pformat=str, asdict=lambda x:{},
                  make_robot_from_config=lambda _:robot, make_default_processors=lambda:(processor,processor,processor),
                  LeRobotDatasetMetadata=lambda **kw:metadata, make_policy=lambda **kw:policy,
                  make_pre_post_processors=lambda **kw:(processor,processor),
                  is_headless=lambda:not normal_exit,
                  init_keyboard_listener=lambda **kw:(listener, {'reset':False,'exit':True}),
                  get_safe_torch_device=lambda *a,**kw:'cpu',
                  time=Clock(), np=NS(array=list))
        # On a headless successful run request exit from the observation stub.
        def observation():
            ns['eval_loop'].__globals__['time'].now += 1
            if not normal_exit:
                raise KeyboardInterrupt()
            return {}
        robot.get_observation.side_effect = observation
        fn = load_function(EVAL, 'eval_loop', ns)
        with patch.dict('sys.modules', {'pynput': NS(keyboard=NS())}):
            if normal_exit:
                fn(cfg)
                listener.stop.assert_called_once()
            else:
                with self.assertRaises(type(connect_error or error) if (connect_error or error) else KeyboardInterrupt):
                    fn(cfg)
        robot.disconnect.assert_called_once()

    def test_eval_disconnects_and_stops_listener_on_normal_exit(self):
        self.run_eval(normal_exit=True)

    def test_eval_disconnects_after_partial_connection_failure(self):
        self.run_eval(connect_error=RuntimeError('partial connection failed'))

    def test_eval_disconnects_on_keyboard_interrupt(self):
        self.run_eval()

    def test_eval_disconnects_on_runtime_error(self):
        self.run_eval(RuntimeError('observation failed'))

    def test_cleanup_error_preserves_original_exception(self):
        with self.assertLogs(level='ERROR'):
            self.run_eval(KeyboardInterrupt(), RuntimeError('disconnect failed'))

    def test_async_eval_counts_observation_time(self):
        tree = ast.parse(STARVLA.read_text())
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
                    and isinstance(n.test, ast.Constant) and n.test.value is True)
        for mode in ('rtc', 'te'):
            for observation_delay in (0.02, 0.05):
                with self.subTest(mode=mode, delay=observation_delay):
                    clock = Clock()
                    events = {'reset':False, 'exit':False}
                    def observe():
                        clock.now += observation_delay
                        return {'camera':None}
                    def send(action):
                        events['exit'] = True
                    wrapper = NS(chunk_index=0, current_step_in_chunk=0, get_action=lambda q:[0]*8)
                    ns = dict(time=clock, events=events, rtc=wrapper if mode=='rtc' else None,
                              te=wrapper if mode=='te' else None, robot=NS(get_observation=observe,send_action=send),
                              cfg=NS(camera_keys=['camera'],single_task='test'), gripper_cmd=None,
                              state_q01=None,state_q99=None,_build_state=lambda *a:None,
                              _build_action_dict=lambda a:a, seam_writer=None,step_writer=None,last_action=None,
                              current_step=0,sleep_time_s=1/30,precise_sleep=clock.sleep)
                    exec(compile(ast.Module(body=[loop],type_ignores=[]),str(STARVLA),'exec'),ns)
                    self.assertAlmostEqual(clock.now, max(1/30, observation_delay))


if __name__ == '__main__':
    unittest.main()
