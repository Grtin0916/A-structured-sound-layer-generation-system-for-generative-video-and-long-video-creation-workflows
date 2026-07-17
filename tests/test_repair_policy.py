import unittest

from soundlayer.repair.repair_policy import compile_policy


class RepairPolicyTest(unittest.TestCase):
    def test_clipping_compiles_to_bounded_attenuation(self):
        plan = compile_policy("clipping")
        self.assertIsNotNone(plan)
        self.assertEqual(plan["action"], "attenuate_limit")
        self.assertLessEqual(plan["parameters"]["gain"], 1)
        self.assertGreater(plan["parameters"]["gain"], 0)
        self.assertTrue(plan["execution_ready"])

    def test_candidate_replace_is_explicitly_blocked(self):
        plan = compile_policy("naive_less_controllable")
        self.assertIsNotNone(plan)
        self.assertEqual(plan["action"], "candidate_replace")
        self.assertFalse(plan["execution_ready"])
        self.assertEqual(plan["blocked_reason"], "deterministic_action_not_available")

    def test_unknown_failure_has_no_policy(self):
        self.assertIsNone(compile_policy("not_real"))


if __name__ == "__main__":
    unittest.main()
