"""Recovery orchestration, counterfactual gate, and prediction-error feedback."""
from dataclasses import dataclass, field, asdict
from .core import State, loss, recovered, observations, transition
from .planner import plan, baseline_action
from .uncertainty import actual_transition, hidden_conditions, prediction_error
from .agent import investigate, AgentReport
from .ml import predict_action_loss

VALID_STRATEGIES = ('reactive', 'counterfactual', 'llm_only', 'llm_guarded')

@dataclass
class RunContext:
    seed: int = 42
    uncertainty: float = 1.0
    risk_memory: dict[str, float] = field(default_factory=dict)


def choose_with_gate(s: State, proposal: str, risk_memory: dict | None = None, margin: float = 18.0):
    """Reject candidate if its best modeled 3-step continuation is much riskier.

    Uses nominal planner and a transparent accumulated surprise penalty. Risk
    memory is a hand-coded adaptation heuristic, not a trained model.
    """
    candidates = plan(s, horizon=3, top_k=250)
    if not candidates or not candidates[0]['actions']:
        return proposal, {'decision':'already_recovered', 'proposed':proposal}
    penalty = risk_memory or {}

    learned_excess = {}
    for action in set(o['actions'][0] for o in candidates):
        nominal_one_step = loss(transition(s, action))
        learned_excess[action] = max(0.0, predict_action_loss(s, action) - nominal_one_step)
    def adjusted(o):
        first=o['actions'][0]
        return round(o['score'] + 2.5 * penalty.get(first,0) + 0.75 * learned_excess[first], 2)
    winner = min(candidates, key=adjusted)
    compatible = [c for c in candidates if c['actions'][0] == proposal]
    if not compatible:
        return winner['actions'][0], {'decision':'override', 'proposed':proposal, 'executed':winner['actions'][0], 'reason':'No valid modeled continuation for proposal'}
    proposed_best = min(compatible, key=adjusted)
    gap = round(adjusted(proposed_best)-adjusted(winner), 2)
    if gap > margin:
        return winner['actions'][0], {
            'decision':'override','proposed':proposal,'executed':winner['actions'][0],
            'proposal_score':adjusted(proposed_best),'safest_score':adjusted(winner),
            'risk_gap':gap,'learned_risk_excess':learned_excess.get(proposal,0.0),'reason':'Proposal has a worse modeled recovery continuation than the safer alternative.'}
    return proposal, {'decision':'approved','proposed':proposal,'executed':proposal,
                      'proposal_score':adjusted(proposed_best),'safest_score':adjusted(winner), 'risk_gap':gap,
                      'learned_risk_excess':learned_excess.get(proposal,0.0),'reason':'Proposal within allowable modeled risk margin.'}


def step(s: State, ctx: RunContext, strategy: str = 'counterfactual', *,
         api_key: str = '', model: str = '',
         base_url: str = '', agent_client=None):
    if strategy not in VALID_STRATEGIES: raise ValueError(f'Invalid strategy: {strategy}')
    if recovered(s): raise ValueError('System already meets recovery criteria')
    ai = None
    if strategy in ('llm_only','llm_guarded'):
        ai = investigate(s, api_key=api_key,model=model,base_url=base_url,client=agent_client)
        proposed = ai.action
    elif strategy == 'reactive':
        proposed = baseline_action(s)
    else:
        proposed = plan(s)[0]['actions'][0]
    if strategy == 'llm_guarded':
        action, gate = choose_with_gate(s, proposed, ctx.risk_memory)
    else:
        action, gate = proposed, {'decision':'not_applied','proposed':proposed,'executed':proposed,
                                  'reason':'This strategy has no counterfactual gate.'}
    initial_loss = loss(s)
    hidden = hidden_conditions(ctx.seed, ctx.uncertainty)
    actual = actual_transition(s, action, hidden,seed=ctx.seed)
    pred = prediction_error(s, action, actual)

    surprise = max(0.0, pred['prediction_error'])
    old = ctx.risk_memory.get(action,0.0)
    ctx.risk_memory[action] = round(0.65 * old + 0.35 * surprise, 2)
    event = {
        'step':s.steps+1, 'strategy':strategy, 'proposed':proposed,'action':action,
        'gate':gate, 'loss_before':initial_loss, **pred,
        'recovered':recovered(actual), 'risk_memory':dict(ctx.risk_memory),
        'ai':asdict(ai) if ai else None,
    }
    return actual,event


def episode(s: State, ctx: RunContext, strategy: str='counterfactual',max_steps:int=6,**kwargs):
    events=[]
    starting_loss=loss(s)
    for _ in range(max_steps):
        if recovered(s): break
        s, event = step(s,ctx,strategy,**kwargs)
        events.append(event)
    losses = [starting_loss] + [e['actual_loss'] for e in events]
    return {'scenario':s.scenario, 'strategy':strategy, 'recovered':recovered(s),
            'steps':len(events),'final_loss':loss(s), 'peak_loss':max(losses),
            'cumulative_loss':round(sum(losses[1:]),2),
            'prediction_mae':round(sum(abs(e['prediction_error']) for e in events)/len(events),2) if events else 0.0,
            'overrides':sum(1 for e in events if e['gate']['decision']=='override'),
            'events':events,'final_state':asdict(s)}
