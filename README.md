# AFTERMATH

Counterfactual recovery planning for interconnected services


## Overview

A recovery operation can solve one failure while creating another. Restarting an overloaded payment service, for example, may release queued retries and overwhelm its database dependency.

AFTERMATH investigates simulated infrastructure incidents and evaluates alternative interventions before executing them. It combines model-assisted diagnosis with counterfactual planning and a safety gate, then measures the difference between predicted and observed results.

The application runs entirely inside a **made-up payment-platform environment**

## System design

```
Simulated logs & metrics
         |
         v
Anomaly detection (IsolationForest)
         |
         v
LLM investigator (configured provider; tool calls)
         |
         v
Proposed intervention
         |
         v
Counterfactual planner + impact model (RandomForest)
         |
         v
Safety gate (approve / override)
         |
         v
Sandbox intervention -> observe -> verify -> replan
```

The **IsolationForest** and **RandomForestRegressor** models are fitted locally on synthetic observations and nominal action transitions. The investigator uses a pretrained, tool-capable language model through an OpenAI-compatible Chat Completions API. The model is not included in the repository and is not fine-tuned here. Model calls require external credentials.

The counterfactual planner is a classical bounded search algorithm, not a trained neural network. It evaluates three-action sequences using cumulative system loss, worst intermediate loss, intervention cost, and a recovery bonus. Only the first selected action is executed before planning again.

Recovery actions are restricted to six simulated operations: payment restart, retry throttling, database pool expansion, queue clearing, authentication restoration, and waiting for background recovery.





### Local Python

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

### Hosted model configuration

The dashboard's sidebar accepts an API key, a model identifier, and a compatible API base URL. The endpoint must support **Chat Completions function/tool calling**. Qwen is the intended model family; the specific checkpoint and hosted provider must be configured separately. Local ML models, planning, and synthetic benchmarks do not require an external API.

API keys are entered through a password input and are not part of the project files. Generated traces should be reviewed before being shared.

## Interface

- **System & recovery:** live synthetic metrics, anomaly detection, ranked alternative futures, and recovery controls.
- **Agent investigation:** model-issued diagnostic tool calls, evidence, proposed intervention, and safety gate decision.
- **Experiment lab:** seeded offline recovery benchmarks and optional LLM-only versus gated-model comparison.
- **Architecture & limitations:** implemented components and validation boundaries.

**Recovery strategies**

| Strategy | Decision mechanism | LLM call |
| --- | --- | --- |
| `reactive` | Fixed symptom-based heuristic | No |
| `counterfactual` | Three-step nominal simulation and ranking | No |
| `llm_only` | Pretrained model investigates and proposes an action | Yes |
| `llm_guarded` | Same investigator, with counterfactual safety gate | Yes |



| Metric | Reactive | Counterfactual |
| --- | ---: | ---: |
| Recovery rate | 68% | 100% |
| Mean actions | 3.68 | 2.68 |
| Mean peak loss | 268.70 | 204.88 |
| Mean cumulative loss | 615.55 | 225.18 |







