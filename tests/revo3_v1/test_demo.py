import unittest

from revo3_v1.demo import DemoConfig, run
from revo3_v1.planner import SupportedTask, TASK_GRASP_PRIMITIVES


class DemoTest(unittest.TestCase):
    def test_full_mock_pipeline_reaches_complete(self):
        result = run(DemoConfig(task=SupportedTask.BOTTLE))
        self.assertEqual(result["terminal_output"], "COMPLETE")
        self.assertEqual(result["chunk_size"], 16)
        self.assertEqual(result["refine_offsets"], [4, 8, 12])
        self.assertGreaterEqual(result["commands_written"], 24)
        self.assertIn("no robot/task-success claim", result["verification_scope"])

    def test_all_allowlisted_tasks_plumb(self):
        for task in SupportedTask:
            with self.subTest(task=task.value):
                result = run(DemoConfig(task=task, emulate_release=False))
                self.assertEqual(result["stable_output"], "HOLD")
                self.assertEqual(result["primitive"], TASK_GRASP_PRIMITIVES[task].value)
                style = TASK_GRASP_PRIMITIVES[task].value.removesuffix("_GRASP").lower()
                self.assertIn(f"{style} grasp", result["instruction"].lower())


if __name__ == "__main__":
    unittest.main()
