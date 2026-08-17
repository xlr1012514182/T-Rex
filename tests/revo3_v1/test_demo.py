import unittest

from revo3_v1.demo import DemoConfig, run
from revo3_v1.planner import SupportedTask


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


if __name__ == "__main__":
    unittest.main()
