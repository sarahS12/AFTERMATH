import unittest
from types import SimpleNamespace as N
from aftermath.core import initial_state, loss, transition
from aftermath.controller import RunContext, step, episode, choose_with_gate
from aftermath.uncertainty import actual_transition, hidden_conditions
from aftermath.ml import assess, trained_detector
from aftermath.agent import investigate


def fake_call(call_id, name, arguments='{}'):
    return N(id=call_id, function=N(name=name, arguments=arguments))


class FakeCompletions:
    def __init__(self): self.calls = 0
    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            calls = [fake_call('1','get_metrics'),fake_call('2','get_logs'),fake_call('3','get_anomaly_signal')]
        else:
            calls = [fake_call('4','submit_recovery_action','{"action":"restart_payments","hypothesis":"payment overload","evidence":"payment errors and retry backlog","reason":"restore service"}')]
        return N(choices=[N(message=N(content=None, tool_calls=calls))],usage=N(prompt_tokens=20,completion_tokens=10))

class FakeClient:
    def __init__(self): self.chat=N(completions=FakeCompletions())

class TestAgentAndSafety(unittest.TestCase):
    def test_isolation_forest_is_genuinely_fitted(self):
        detector=trained_detector()
        self.assertTrue(hasattr(detector,'estimators_'))
        self.assertTrue(assess(initial_state('retry_storm'))['anomaly'])

    def test_hidden_execution_is_different_from_nominal(self):
        s=initial_state('retry_storm')
        actual=actual_transition(s,'restart_payments',hidden_conditions(42,1),seed=42)
        nominal=transition(s,'restart_payments')
        self.assertNotEqual(loss(actual),loss(nominal))
        zero=actual_transition(s,'restart_payments',hidden_conditions(42,0),seed=42)
        self.assertEqual(loss(zero),loss(nominal))

    def test_llm_agent_really_requests_tools_then_proposes_action(self):
        client=FakeClient()
        out=investigate(initial_state('retry_storm'),api_key='',client=client,model='mock-for-test')
        self.assertEqual(out.action,'restart_payments')
        self.assertEqual(client.chat.completions.calls,2)
        self.assertEqual([x['tool'] for x in out.trace],['get_metrics','get_logs','get_anomaly_signal','submit_recovery_action'])

    def test_no_credentials_no_fake_llm_action(self):
        with self.assertRaises(ValueError):
            investigate(initial_state('retry_storm'),api_key='')

    def test_gate_overrides_risky_llm_action(self):
        s=initial_state('retry_storm')
        act, verdict=choose_with_gate(s,'restart_payments')
        self.assertEqual(verdict['decision'],'override')
        self.assertNotEqual(act,'restart_payments')

    def test_llm_only_and_guarded_act_differently_same_mock_model(self):
        s=initial_state('retry_storm')
        _s, untreated=step(s,RunContext(seed=42),strategy='llm_only',agent_client=FakeClient(),model='mock-test-model')
        _s, guarded=step(s,RunContext(seed=42),strategy='llm_guarded',agent_client=FakeClient(),model='mock-test-model')
        self.assertEqual(untreated['action'],'restart_payments')
        self.assertNotEqual(guarded['action'],'restart_payments')
        self.assertGreater(untreated['actual_loss'],guarded['actual_loss'])
        self.assertEqual(guarded['ai']['model'],'mock-test-model')

    def test_experiment_reproducible_across_hidden_seeds(self):
        a=episode(initial_state('mixed_failure'),RunContext(seed=38,uncertainty=1.0),'counterfactual')
        b=episode(initial_state('mixed_failure'),RunContext(seed=38,uncertainty=1.0),'counterfactual')
        self.assertEqual([e['actual_loss'] for e in a['events']], [e['actual_loss'] for e in b['events']])
        self.assertTrue(a['prediction_mae'] > 0)

if __name__=='__main__': unittest.main()
