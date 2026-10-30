import unittest
from aftermath.core import initial_state, transition, loss, recovered
from aftermath.planner import plan
from aftermath.controller import episode, RunContext


class TestAftermath(unittest.TestCase):
    def test_restart_can_cause_secondary_overload(self):
        s = initial_state("retry_storm")
        after = transition(s, "restart_payments")
        self.assertGreater(after.db_load, s.db_load)
    def test_throttling_limits_restart_surge(self):
        s = initial_state("retry_storm")
        direct = transition(s, "restart_payments")
        controlled = transition(transition(s, "throttle_retries"), "restart_payments")
        self.assertLess(controlled.db_load, direct.db_load)
    def test_planner_only_allowed_actions(self):
        self.assertTrue(all(a in ["restart_payments", "throttle_retries", "expand_db_pool", "clear_queue", "restore_auth_config", "wait"] for p in plan(initial_state("retry_storm")) for a in p["actions"]))
    def test_healthy_is_recovered(self):
        self.assertTrue(recovered(initial_state("healthy")))
    def test_execution_returns_history(self):
        result = episode(initial_state("retry_storm"), RunContext(seed=42), "counterfactual")
        self.assertGreater(len(result["events"]), 0)

if __name__ == "__main__": unittest.main()
