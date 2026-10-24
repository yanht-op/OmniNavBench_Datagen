"""Collection input and media admission tests; no Isaac Sim required."""
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from datagen.cli import parse_args, prepare_inputs
from datagen.training_samples import build_manifest, validate_sample
from datagen.batch import discover_scenes
from datagen.trajectory_targets import future_targets
from datagen.generation.endpoints import sample_endpoint_path, route_instruction


class CollectionTests(unittest.TestCase):
    def test_runtime_diagnostics_cleanup_after_success_and_failure(self):
        from datagen.runtime import collect
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(output=Path(directory))
                with patch("datagen.runtime._collect", side_effect=RuntimeError("test failure") if fail else None,
                           return_value=[]) as inner, \
                     patch("datagen.runtime.faulthandler.dump_traceback_later") as start, \
                     patch("datagen.runtime.faulthandler.cancel_dump_traceback_later") as cancel, \
                     contextlib.redirect_stdout(io.StringIO()):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "test failure"):
                            collect(args, None, None)
                    else:
                        self.assertEqual(collect(args, None, None), [])
                    start.assert_called_once()
                    self.assertEqual(start.call_args.args, (120,))
                    self.assertFalse(start.call_args.kwargs["exit"])
                    cancel.assert_called_once()
                    self.assertTrue(start.call_args.kwargs["file"].closed)
                    inner.assert_called_once()
                content = (args.output / "python_stacks.log").read_text()
                self.assertIn("Python stacks every 120", content)
                if fail:
                    self.assertIn("RuntimeError: test failure", content)

    def test_runtime_stage_records_failure_without_swallowing_it(self):
        from datagen.runtime import _phase
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = SimpleNamespace(output=Path(directory))
            def interrupted():
                raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                _phase(args, "physics initialization", interrupted)
            content = (args.output / "progress.log").read_text()
            self.assertIn("physics initialization: begin", content)
            self.assertIn("KeyboardInterrupt", content)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def sample(self, depth_frames=3):
        directory = self.root / "episode_000000"
        (directory / "path").mkdir(parents=True)
        (directory / "video").mkdir()
        recording = {"gt_path": [
            {"frame": i, "time_s": (i + 1) / 15, "xyz": [i / 10, 0, 0], "yaw_deg": 0}
            for i in range(3)]}
        scenario = {"id": "generated_0", "scene": {"units_in_meters": 1}, "recording": recording,
                    "task": {"navigation": {"instruction": "Go to the chair."}}}
        (directory / "episode.json").write_text(json.dumps({"scenarios": [scenario]}))
        (directory / "path/path.json").write_text(json.dumps(recording))
        (directory / "attempt.json").write_text(json.dumps({"status": "succeeded"}))
        for name, count in (("rgb", 3), ("depth", depth_frames)):
            writer = cv2.VideoWriter(str(directory / f"video/{name}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 15, (32, 24))
            self.assertTrue(writer.isOpened())
            for i in range(count):
                writer.write(np.full((24, 32, 3), i * 40, dtype=np.uint8))
            writer.release()
        return directory

    def test_real_video_and_trajectory_are_admitted(self):
        directory = self.sample()
        result = validate_sample(directory)
        self.assertEqual(result["frames"], 3)
        self.assertEqual(result["fps"], 15)
        summary = build_manifest(self.root, [SimpleNamespace(output_dir=directory)], requested=1)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(len((self.root / "manifest.jsonl").read_text().splitlines()), 1)

    def test_short_depth_video_is_excluded(self):
        directory = self.sample(depth_frames=2)
        summary = build_manifest(self.root, [SimpleNamespace(output_dir=directory)], requested=1)
        self.assertEqual(summary["accepted"], 0)
        self.assertEqual(len(summary["validation_failures"]), 1)
        self.assertEqual((self.root / "manifest.jsonl").read_text(), "")

    def test_sidecar_mismatch_rejected(self):
        directory = self.sample()
        (directory / "path/path.json").write_text('{"gt_path": []}')
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_sample(directory)

    def test_repeated_timestamp_rejected(self):
        directory = self.sample()
        payload = json.loads((directory / "episode.json").read_text())
        recording = payload["scenarios"][0]["recording"]
        recording["gt_path"][1]["time_s"] = recording["gt_path"][0]["time_s"]
        (directory / "episode.json").write_text(json.dumps(payload))
        (directory / "path/path.json").write_text(json.dumps(recording))
        with self.assertRaisesRegex(ValueError, "increase"):
            validate_sample(directory)

    def inputs(self):
        (self.root / "scene.usd").touch()
        (self.root / "robot.usd").touch()
        config = self.root / "config.yaml"
        config.write_text("simulator:\n  physics_dt: 0.033333333333\n")
        source = self.root / "source.json"
        scenario = {"id": "source", "scene": {"usd_path": "scene.usd", "units_in_meters": 1},
                    "robots": {"entries": [{"type": "carter_v1", "usd_path": "robot.usd", "rb_gt_waypoints": [1],
                        "initial_pose": {"position": [0, 0, 0.3], "orientation_deg": 0}}]},
                    "task": {"subtasks": [{"type": "EQA"}], "navigation": {"answer": "old"}},
                    "recording": {"gt_path": [1]}, "answer": "old", "virtual_humans": {"count": 1}}
        source.write_text(json.dumps({"scenarios": [scenario]}))
        argv = ["--config", str(config), "--envset", str(source), "--scene-root", str(self.root),
                "--output", str(self.root / "output")]
        return source, argv

    def test_source_labels_removed_without_mutating_source(self):
        source, argv = self.inputs()
        original = source.read_bytes()
        scenario, cfg = prepare_inputs(parse_args(argv))
        self.assertEqual(scenario["task"], {"navigation": {}})
        self.assertNotIn("recording", scenario)
        self.assertNotIn("rb_gt_waypoints", scenario["robots"]["entries"][0])
        self.assertNotIn("virtual_humans", scenario)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(cfg.task.types, ["ObjectNav"])
        self.assertFalse((self.root / "output").exists())

    def test_existing_output_reused_without_modifying_contents(self):
        _, argv = self.inputs()
        (self.root / "output").mkdir()
        marker = self.root / "output/keep.txt"
        marker.write_text("keep")
        prepare_inputs(parse_args(argv))
        self.assertEqual(marker.read_text(), "keep")

    def test_output_file_refused(self):
        _, argv = self.inputs()
        (self.root / "output").write_text("keep")
        with self.assertRaisesRegex(ValueError, "must be a directory"):
            prepare_inputs(parse_args(argv))

    def test_repeated_cli_runs_isolate_diagnostics_and_keep_root_manifest(self):
        from datagen.cli import main
        _, argv = self.inputs()
        output = self.root / "output"
        output.mkdir()
        manifest = output / "manifest.jsonl"
        manifest.write_text('{"sample_id":"existing"}\n')

        def collect(args, config, source):
            self.assertTrue(source.is_file())
            self.assertEqual(config.output_dir.parent, args.output)
            config.output_dir.mkdir()
            return []

        with patch("datagen.cli.importlib.util.find_spec", return_value=object()), \
             patch("datagen.runtime.collect", side_effect=collect), \
             contextlib.redirect_stdout(io.StringIO()):
            # Empty mocked collections report failure, but their diagnostics must coexist.
            self.assertEqual(main(argv), 2)
            self.assertEqual(main(argv), 2)
        runs = list((output / "_runs").iterdir())
        self.assertEqual(len(runs), 2)
        for run in runs:
            self.assertTrue((run / "summary.json").is_file())
        self.assertEqual(manifest.read_text(), '{"sample_id":"existing"}\n')

    def test_navmesh_override_converts_meters_without_editing_source(self):
        source, argv = self.inputs()
        payload = json.loads(source.read_text())
        payload["scenarios"][0]["scene"]["units_in_meters"] = 0.01
        payload["scenarios"][0]["navmesh"] = {"agent_radius": 10, "max_step_height": 30}
        source.write_text(json.dumps(payload))
        original = source.read_bytes()
        scenario, _ = prepare_inputs(parse_args(argv + ["--navmesh-agent-radius-m", "0.35",
                                                        "--navmesh-max-step-height-m", "0.05"]))
        self.assertAlmostEqual(scenario["navmesh"]["agent_radius"], 35)
        self.assertAlmostEqual(scenario["navmesh"]["max_step_height"], 5)
        self.assertEqual(source.read_bytes(), original)

    def test_fps_above_physics_rate_refused(self):
        _, argv = self.inputs()
        with self.assertRaisesRegex(ValueError, "frequency"):
            prepare_inputs(parse_args(argv + ["--fps", "60"]))

    def test_invalid_sampling_ranges_refused(self):
        _, argv = self.inputs()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(argv + ["--object-goal-min-m", "3", "--object-goal-max-m", "1"])

    def test_scene_dedup_and_matterport_exclusion(self):
        root = self.root / "annotations/train/original/car/scene_a"
        root.mkdir(parents=True)
        scenarios = [{"id": "one", "scene": {"category": "GRScenes", "usd_path": "a.usd"}},
                     {"id": "two", "scene": {"category": "GRScenes", "usd_path": "a.usd"}},
                     {"id": "mp", "scene": {"category": "MP3D", "usd_path": "mp.usd"}}]
        (root / "episode.json").write_text(json.dumps({"scenarios": scenarios}))
        scenes = discover_scenes(self.root)
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["asset"], "a.usd")

    def test_future_local_coordinates_and_end_mask(self):
        points = [{"frame": i, "time_s": i, "xyz": [0, i*100, 0], "yaw_deg": 90} for i in range(3)]
        targets = list(future_targets(points, 0.01, (0.5, 1.0)))
        np.testing.assert_allclose(targets[0]["future_xy_local_m"], [[0.5, 0], [1, 0]], atol=1e-8)
        self.assertEqual(targets[-1]["valid"], [False, False])
        self.assertTrue(targets[-1]["is_last_observation"])

    def test_endpoint_sampling_rejects_disconnected_and_wrong_lengths(self):
        goals = iter([[1, 0, 0], [0.2, 0, 0], [3, 0, 0]])
        class Geometry:
            def sample_random_point(self):
                return next(goals)
            def query_path(self, start, goal):
                return None if goal[0] == 1 else [start, goal]
            def get_clearance(self, goal):
                return 1
        points, length = sample_endpoint_path(Geometry(), [0, 0, 0], units=1,
                                               min_length_m=2, max_length_m=4, attempts=3)
        self.assertAlmostEqual(length, 3)
        self.assertAlmostEqual(points[-1][0], 3)

    def test_route_language_matches_turn_direction(self):
        text = route_instruction([[0, 0, 0], [2, 0, 0], [2, 1, 0]], 0, 1)
        self.assertIn("2.00 meters", text)
        self.assertIn("left approximately 90 degrees", text)
        self.assertTrue(text.endswith("Stop at the destination."))

    def test_pointnav_mode_does_not_require_semantic_visibility(self):
        _, argv = self.inputs()
        _, cfg = prepare_inputs(parse_args(argv + ["--task", "pointnav", "--path-min-m", "3", "--path-max-m", "6"]))
        self.assertTrue(cfg.task.sample_endpoints)
        self.assertEqual(cfg.min_pixels_visible, 0)
        self.assertEqual(cfg.task.path_max_m, 6)


if __name__ == "__main__":
    unittest.main()
