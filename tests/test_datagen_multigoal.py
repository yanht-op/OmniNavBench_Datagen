"""Multi-object sampler, real media validation, and standard dataset export regressions."""
import copy
import contextlib
import io
import json
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from datagen.core.registry import ObjectEntry
from datagen.depth import save_depth_frame
from datagen.generation.blueprint import ChainSampler
from datagen.export.exporter import EpisodeExporter
from datagen.export.standard import publish_samples
from datagen.training_samples import build_manifest, validate_sample
from OmniNavExt.envset.recording import build_recording_payload


class MultiGoalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def episode(self):
        work = self.root / "work"
        directory = work / "episode_000000"
        (directory / "video").mkdir(parents=True)
        points = [{"frame": i, "sim_step": i+1, "time_s": (i+1)/10,
                   "xyz": [i*.1, 0., .25], "yaw_deg": 0., "distance_xy": .1 if i else 0.,
                   "distance_total_xy": i*.1} for i in range(6)]
        segments = []
        for i, category in enumerate(["chair", "table", "sofa"]):
            start, end = 2*i, 2*i+1
            segments.append({"step": i, "target": {"object_id": f"id{i}", "category": category,
                             "position": [i, 1, 0.5], "prim_path": f"/World/{category}"},
                             "instruction": f"Go to the {category}.",
                             "start_capture_frame": start, "end_capture_frame": end,
                             "start_time_s": points[start]["time_s"], "end_time_s": points[end]["time_s"],
                             "achieved_position": points[end]["xyz"]})
        instruction = " ".join(s["instruction"] for s in segments)
        recording = build_recording_payload(instruction=instruction, gt_path=points,
            metadata={"task_types": ["ObjectNav"], "requested_subtasks": 3,
                      "distinct_object_targets": True, "distinct_object_categories": True, "raw_depth": True})
        for name in ("rgb", "depth"):
            writer = cv2.VideoWriter(str(directory / "video" / f"{name}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
            self.assertTrue(writer.isOpened())
            for i in range(6):
                writer.write(np.full((24, 32, 3), i*20, np.uint8))
            writer.release()
        for i in range(6):
            save_depth_frame(directory / "video/depth_raw", i, np.full((24, 32), i+.123456, np.float32))
        episode = SimpleNamespace(episode_id="0", output_dir=directory, instruction=instruction,
            recording=recording, metadata={"task_types": ["ObjectNav"], "object_segments": segments,
                "target": segments[-1]["target"], "start_pose": {"position": [0, 0, .25], "orientation_deg": 0}})
        template = {"id": "source", "scene": {"category": "GRScenes", "units_in_meters": 1},
                    "robots": {"entries": [{"type": "carter_v1", "usd_path": "robot.usd"}]}, "task": {}}
        EpisodeExporter().export(episode=episode, template_scenario=template)
        return work, directory, episode

    def test_three_goal_standard_export_and_training_alignment(self):
        work, directory, episode = self.episode()
        summary = build_manifest(work, [episode], requested=1)
        self.assertEqual(summary["accepted"], 1)
        dataset = self.root / "dataset"
        dataset.mkdir()
        result = publish_samples(work, dataset, scene_id="GRScene_test", robot="car", summary=summary,
                                 manifest_path=dataset / "manifest.jsonl")
        self.assertEqual(result["status"], "complete")
        annotation = dataset / "annotations/train/original/car/GRScene_test/final_episode_0.json"
        scenario = json.loads(annotation.read_text())["scenarios"][0]
        self.assertEqual([s["object_id"] for s in scenario["task"]["subtasks"]], ["id0", "id1", "id2"])
        self.assertEqual(scenario["robots"]["entries"][0]["rb_gt_waypoints"], scenario["recording"]["gt_path"])
        self.assertEqual(scenario["task"]["navigation"]["goal_position"], [.5, 0, .25])
        media = dataset / "videos/train/car/GRScene_test/final_episode_0"
        self.assertTrue((media / "rgb.mp4").is_file())
        self.assertTrue((media / "depth.mp4").is_file())
        np.testing.assert_array_equal(np.load(media / "depth_raw/000003.npy"), np.load(directory / "video/depth_raw/000003.npy"))
        targets = [json.loads(l) for l in (media / "trajectory_targets.jsonl").read_text().splitlines()]
        self.assertEqual([t["subtask_index"] for t in targets], [0, 0, 1, 1, 2, 2])
        self.assertEqual([t["subtask_end"] for t in targets], [False, True, False, True, False, True])
        before = annotation.read_bytes()
        result = publish_samples(work, dataset, scene_id="GRScene_test", robot="car", summary=summary,
                                 manifest_path=dataset / "retry_manifest.jsonl")
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(annotation.read_bytes(), before)
        self.assertTrue(annotation.with_name("final_episode_1.json").is_file())
        records = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual(len({record["sample_id"] for record in records}), 2)
        self.assertEqual(len((dataset / "retry_manifest.jsonl").read_text().splitlines()), 1)

    def test_append_skips_existing_media_and_annotation_numbers(self):
        work, directory, episode = self.episode()
        summary = build_manifest(work, [episode], requested=1)
        dataset = self.root / "dataset"
        media = dataset / "videos/train/car/GRScene_test/final_episode_8"
        media.mkdir(parents=True)
        (media / "keep.txt").write_text("interrupted run")
        annotations = dataset / "annotations/train/original/car/GRScene_test"
        annotations.mkdir(parents=True)
        (annotations / "final_episode_3.json").write_text("existing")
        result = publish_samples(work, dataset, scene_id="GRScene_test", robot="car", summary=summary,
                                 manifest_path=dataset / "manifest.jsonl")
        self.assertEqual(result["accepted"], 1)
        self.assertTrue((annotations / "final_episode_9.json").is_file())
        self.assertEqual((annotations / "final_episode_3.json").read_text(), "existing")
        self.assertEqual((media / "keep.txt").read_text(), "interrupted run")

    def test_incomplete_chain_and_raw_depth_are_rejected(self):
        _, directory, _ = self.episode()
        payload = json.loads((directory / "episode.json").read_text())
        original = copy.deepcopy(payload)
        payload["scenarios"][0]["datagen_segments"].pop()
        (directory / "episode.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_sample(directory)
        (directory / "episode.json").write_text(json.dumps(original))
        (directory / "video/depth_raw/000002.npy").unlink()
        with self.assertRaisesRegex(ValueError, "Raw depth frames"):
            validate_sample(directory)

    def test_frame_overlap_rejected(self):
        _, directory, _ = self.episode()
        payload = json.loads((directory / "episode.json").read_text())
        payload["scenarios"][0]["datagen_segments"][1]["start_capture_frame"] = 1
        (directory / "episode.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "noncontiguous"):
            validate_sample(directory)

    def test_sampler_chains_distinct_targets_and_categories(self):
        objects = [ObjectEntry(str(i), f"/World/{i}", category, np.array([i*3+3., 1., 0.5]))
                   for i, category in enumerate(["chair", "chair", "table", "sofa"])]
        registry = SimpleNamespace(query=lambda: objects)
        geometry = SimpleNamespace(query_path=lambda a, b: [a, b], snap_point=lambda p: np.asarray(p),
                                   get_clearance=lambda p: 1.)
        sampler = ChainSampler(registry, geometry, rng=random.Random(42))
        chain = sampler.sample_chain(3, ["ObjectNav"], start_pos=np.array([0., 0., 0.]),
                                     distinct_object_targets=True, distinct_object_categories=True)
        self.assertEqual(len(chain.nodes), 3)
        self.assertEqual(len({n.target_object.category for n in chain.nodes}), 3)
        for a, b in zip(chain.nodes, chain.nodes[1:]):
            np.testing.assert_array_equal(a.end_pos, b.start_pos)

    def test_batch_plan_preserves_scene_and_standard_dataset_root(self):
        from datagen.batch import main
        source = self.root / "source/annotations/train/original/car/GRScene_test"
        source.mkdir(parents=True)
        payload = {"scenarios": [{"id": "source_episode", "scene": {
            "category": "GRScenes", "usd_path": "GRScenes/test.usd",
            "units_in_meters": 1}}]}
        for name in ("final_episode_0.json", "final_episode_1.json"):
            (source / name).write_text(json.dumps(payload))
        output = self.root / "generated"
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(["--dataset-root", str(self.root / "source"),
                         "--scene-root", str(self.root / "assets"),
                         "--config", "configs/carter_v1_test.yaml", "--output", str(output),
                         "--num-subtasks", "4", "--plan-only"])
        self.assertEqual(code, 0)
        plan = json.loads(stream.getvalue())
        self.assertEqual(len(plan), 1)
        command = plan[0]["command"]
        for flag, expected in (("--task", "objectnav"), ("--num-subtasks", "4"),
                               ("--scene-id", "GRScene_test"), ("--dataset-output", str(output))):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertFalse(output.exists())
        output.mkdir()
        root_manifest = output / "manifest.jsonl"
        root_manifest.write_text('{"sample_id":"existing"}\n')
        with patch("datagen.batch.subprocess.run", return_value=SimpleNamespace(returncode=2)), \
             contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2):
                self.assertEqual(main(["--dataset-root", str(self.root / "source"),
                                       "--scene-root", str(self.root / "assets"),
                                       "--config", "configs/carter_v1_test.yaml",
                                       "--output", str(output)]), 2)
        self.assertEqual(len(list((output / "_runs").iterdir())), 2)
        self.assertEqual(root_manifest.read_text(), '{"sample_id":"existing"}\n')


if __name__ == "__main__":
    unittest.main()
