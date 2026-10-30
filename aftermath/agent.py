
"""Language-model incident investigator with validated diagnostic tools."""
from dataclasses import dataclass, field
import json
from typing import Any
from .core import ACTIONS, State, observations
from .ml import assess

RUNBOOK = {
    'restart_payments': 'Restarts the payments process. CAUTION: pending retries can overwhelm the database.',
    'throttle_retries': 'Limits retry pressure. Can reduce payment load and the queued retry backlog.',
    'expand_db_pool': 'Increases the nominal database connection capacity. Effectiveness may vary.',
    'clear_queue': 'Drains pending notifications but may briefly increase database work.',
    'restore_auth_config': 'Restores known-good authentication configuration.',
    'wait': 'Allows retry and queue loads to settle. Has no immediate aggressive intervention.',
}

TOOLS = [
    {'type': 'function', 'function': {'name': 'get_metrics', 'description': 'Inspect latest service metrics.', 'parameters': {'type':'object', 'properties':{}, 'additionalProperties':False}}},
    {'type': 'function', 'function': {'name': 'get_logs', 'description': 'Inspect current event logs and warnings.', 'parameters': {'type':'object', 'properties':{}, 'additionalProperties':False}}},
    {'type': 'function', 'function': {'name': 'get_dependencies', 'description': 'Inspect service dependency information.', 'parameters': {'type':'object', 'properties':{}, 'additionalProperties':False}}},
    {'type': 'function', 'function': {'name': 'check_runbook', 'description': 'Read an allowed intervention runbook including side-effect warnings.', 'parameters': {'type':'object','properties':{'action':{'type':'string','enum':ACTIONS}},'required':['action'], 'additionalProperties':False}}},
    {'type': 'function', 'function': {'name': 'get_anomaly_signal', 'description': 'Read output of trained IsolationForest anomaly detector.', 'parameters': {'type':'object','properties':{},'additionalProperties':False}}},
    {'type': 'function', 'function': {'name': 'submit_recovery_action', 'description': 'PROPOSE ONE allowed action with a hypothesis and evidence. Does NOT execute anything. Call once when ready.', 'parameters': {'type':'object', 'properties': {'action': {'type':'string','enum':ACTIONS}, 'hypothesis':{'type':'string'}, 'evidence':{'type':'string'}, 'reason':{'type':'string'}}, 'required':['action','hypothesis','evidence','reason'],'additionalProperties':False}}},
]

SYSTEM_PROMPT = '''You are AFTERMATH's AI incident investigator working in an explicitly SYNTHETIC sandbox.
Investigate before proposing a recovery action. You MUST use get_metrics and get_logs first;
use other read-only tools as needed. Propose exactly one permitted action by calling
submit_recovery_action with concise, evidence-grounded fields. No invented telemetry.
Do not assume one root cause explains every concurrent issue. Do not try to run shell
commands or access actual servers. Your proposal may be rejected by a separate risk gate.
Never imply hypothetical simulations are real incidents. If unsure, propose 'wait' and
state what information is missing. Weigh secondary failure risk, especially retries.
'''

@dataclass
class AgentReport:
    action: str
    hypothesis: str
    evidence: str
    reason: str
    model: str
    trace: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


def _tool_result(name: str, args: dict, state: State):
    observed = observations(state)
    if name == 'get_metrics': return observed['metrics']
    if name == 'get_logs': return {'logs': observed['logs']}
    if name == 'get_dependencies': return {'dependencies': observed['dependencies']}
    if name == 'get_anomaly_signal': return assess(state)
    if name == 'check_runbook':
        a = args.get('action')
        return {'action':a, 'guidance':RUNBOOK[a]} if a in RUNBOOK else {'error':'unknown action'}
    return {'error': 'Unknown read-only tool'}


def investigate(state: State, *, api_key: str, model: str = '',
                base_url: str = '', client: Any = None,
                max_rounds: int = 7) -> AgentReport:
    """Inspect incident telemetry through model-issued tool calls."""
    if client is None and not all((api_key, model, base_url)):
        raise ValueError('Live model calls require an API key, model ID, and provider API base URL.')
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url.rstrip('/'), timeout=35.0, max_retries=1)
    messages = [
        {'role':'system','content':SYSTEM_PROMPT},
        {'role':'user','content':('Investigate the current incident. Use telemetry tools, then propose ONE recovery action using submit_recovery_action. The environment is synthetic; do not assume hidden dynamics are known. ' + 'Initial anomaly-detector signal (not a probability or root-cause diagnosis): ' + json.dumps({'anomaly': assess(state)['anomaly'], 'score': assess(state)['anomaly_score']}))},
    ]
    trace = []
    usage = {'prompt_tokens':0, 'completion_tokens':0}
    called = set()
    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice='auto',
            max_completion_tokens=600
        )
        if getattr(response, 'usage', None):
            usage['prompt_tokens'] += getattr(response.usage, 'prompt_tokens', 0) or 0
            usage['completion_tokens'] += getattr(response.usage, 'completion_tokens', 0) or 0
        msg = response.choices[0].message
        tool_calls = getattr(msg, 'tool_calls', None) or []
        if not tool_calls:
            messages.append({'role':'assistant', 'content':getattr(msg,'content',None) or ''})
            messages.append({'role':'user','content':'Please use the provided function tools to inspect evidence and then submit_recovery_action. Do not answer in free text.'})
            continue
        calls_for_history = []
        for call in tool_calls:
            calls_for_history.append({'id':call.id,'type':'function', 'function':{'name':call.function.name,'arguments':call.function.arguments}})
        messages.append({'role':'assistant', 'content':getattr(msg,'content',None), 'tool_calls':calls_for_history})
        proposal = None
        verified_from_previous_round = {'get_metrics','get_logs'}.issubset(called)
        for call in tool_calls:
            name = call.function.name
            args = {}
            try:
                args = json.loads(call.function.arguments or '{}')
                if not isinstance(args, dict): raise ValueError('Tool args must be an object')
            except (ValueError, TypeError) as exc:
                result = {'error':f'Invalid tool arguments: {exc}'}
            else:
                if name == 'submit_recovery_action':
                    if not verified_from_previous_round:
                        result = {'error':'Inspect BOTH get_metrics and get_logs before proposing recovery.'}
                    elif args.get('action') not in ACTIONS:
                        result = {'error':'The proposed action is not allowed.'}
                    else:
                        proposal = args
                        result = {'accepted_as_proposal':True, 'note':'Not executed. Separate safety gate decides whether to allow this intervention.'}
                elif name in {'get_metrics','get_logs','get_dependencies','get_anomaly_signal','check_runbook'}:
                    result = _tool_result(name, args, state)
                    called.add(name)
                else:
                    result = {'error':'Tool not permitted'}
            trace.append({'tool':name,'arguments':args if isinstance(args,dict) else {},'result':result})
            messages.append({'role':'tool', 'tool_call_id':call.id,'content':json.dumps(result, default=str)})
        if proposal is not None:
            return AgentReport(
                action=proposal['action'], hypothesis=str(proposal.get('hypothesis',''))[:600],
                evidence=str(proposal.get('evidence',''))[:600], reason=str(proposal.get('reason',''))[:600],
                model=model,trace=trace,usage=usage
            )
    raise RuntimeError('The model did not submit a permitted recovery action within the investigation budget. No action executed.')
