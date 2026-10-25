"""Per-step EEF telemetry and dependency-free SVG curves for ordinary eval."""
import csv
import json
import logging
from pathlib import Path
import tempfile
import time

import numpy as np


DIMS = ('x_mm', 'y_mm', 'z_mm', 'r11', 'r21', 'r31', 'r12', 'r22', 'r32', 'gripper')
FIELDS = ('episode', 'step', 'elapsed_s', 'feedback_elapsed_s', 'gripper_feedback_elapsed_s',
          'dry_run', 'temporal_ensemble', 'chunk', 'chunk_offset', 'inference_ms',
          *('action_' + key for key in DIMS), *('measured_' + key for key in DIMS),
          *(f'joint_{i}_rad' for i in range(1, 8)), 'command_json', 'dispatch_json', 'feedback_error')


class EEFActionLog:
    def __init__(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.folder = Path(tempfile.mkdtemp(prefix=time.strftime('eef_eval_%Y%m%d_%H%M%S_'), dir=root))
        self.stream = (self.folder / 'actions.csv').open('w', newline='', buffering=1)
        self.writer = csv.DictWriter(self.stream, fieldnames=FIELDS)
        self.writer.writeheader()
        self.started = time.monotonic()
        logging.info('EEF action logs: %s', self.folder.resolve())

    def feedback(self, robot, cfg, state_reader):
        # Cache only: no camera acquisition or extra controller RPC. Telemetry
        # failures leave an explicit gap rather than interrupting policy control.
        try:
            obs = robot.get_motion_observation()
            now = time.monotonic()
            return dict(
                state=state_reader(robot, obs, cfg),
                joints=[obs[f'{robot.prefix}J{i}.pos'] for i in range(1, 8)],
                feedback_elapsed_s=obs.get('_state_sample_monotonic_s', now) - self.started,
                gripper_feedback_elapsed_s=obs.get('_gripper_sample_monotonic_s', now) - self.started)
        except Exception as exc:
            return dict(feedback_error=str(exc))

    def record(self, *, episode, step, action, command, command_time, feedback, cfg,
               chunk, offset, inference_ms, robot):
        predicted = np.asarray(action, dtype=float).copy()
        predicted[:3] *= 1000  # Canonical action metres -> measured-state mm.
        row = dict(episode=episode + 1, step=step, elapsed_s=command_time - self.started,
                   dry_run=cfg.dry_run, temporal_ensemble=cfg.temporal_ensemble,
                   chunk='' if cfg.temporal_ensemble else chunk,
                   chunk_offset='' if cfg.temporal_ensemble else offset,
                   inference_ms='' if cfg.temporal_ensemble else inference_ms,
                   command_json=json.dumps(command), feedback_error=feedback.get('feedback_error', ''))
        row.update(zip(('action_' + key for key in DIMS), map(float, predicted)))
        if 'state' in feedback:
            row.update(zip(('measured_' + key for key in DIMS), map(float, feedback['state'])))
            row.update(zip((f'joint_{i}_rad' for i in range(1, 8)), map(float, feedback['joints'])))
            row.update({key: feedback[key] for key in ('feedback_elapsed_s', 'gripper_feedback_elapsed_s')})
        reader = getattr(robot, 'get_action_diagnostics', None)
        if not cfg.dry_run and callable(reader):
            try:
                row['dispatch_json'] = json.dumps(reader())
            except Exception:
                logging.warning('Could not read EEF command dispatch telemetry', exc_info=True)
        self.writer.writerow(row)

    def close(self):
        self.stream.close()
        # CSV remains usable even if plotting fails or the process was killed.
        try:
            with (self.folder / 'actions.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for episode in sorted({row['episode'] for row in rows}, key=int):
                self._plot(episode, [row for row in rows if row['episode'] == episode])
        except Exception:
            logging.warning('Could not render EEF action curves; CSV is saved', exc_info=True)

    def _plot(self, episode, rows):
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="1150" viewBox="0 0 1100 1150">',
               '<rect width="1100" height="1150" fill="white"/>',
               f'<text x="30" y="25">EEF episode {episode}: action (blue), feedback before command (orange). '
               f'Dry run: {rows[0]["dry_run"]}</text>']
        for i, dim in enumerate(DIMS):
            left, top = 70 + (i % 2) * 550, 65 + (i // 2) * 220
            series = []
            for prefix, time_key in (('action_', 'elapsed_s'), ('measured_',
                    'gripper_feedback_elapsed_s' if dim == 'gripper' else 'feedback_elapsed_s')):
                series.append([(float(row[time_key]), float(row[prefix + dim]))
                               if row[time_key] and row[prefix + dim] else None for row in rows])
            points = [p for line in series for p in line if p is not None]
            xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
            ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
            dx, dy = max(xmax - xmin, .001), max(ymax - ymin, .001)
            svg += [f'<text x="{left}" y="{top-12}">{dim}</text>',
                    f'<path d="M{left} {top}v150h440" fill="none" stroke="#aaa"/>',
                    f'<text x="{left}" y="{top+172}">{xmin:.2f} — {xmax:.2f} s</text>',
                    f'<text x="{left}" y="{top+190}">range {ymin:.4g} — {ymax:.4g}</text>']
            for line, color in zip(series, ('#2563eb', '#ea580c')):
                # Separate paths at missing feedback samples, preserving gaps.
                path, connected = [], False
                for p in line:
                    if p is None:
                        connected = False
                        continue
                    x, y = left + (p[0]-xmin)/dx*440, top + 150 - (p[1]-ymin)/dy*145
                    path.append(f'{"L" if connected else "M"}{x:.2f},{y:.2f}')
                    svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.5" fill="{color}"/>')
                    connected = True
                svg.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        svg.append('</svg>')
        (self.folder / f'episode_{int(episode):03d}_curves.svg').write_text('\n'.join(svg))
