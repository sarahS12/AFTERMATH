"""AFTERMATH held-out-seed evaluation: synthetic scenarios, no API calls."""
import pandas as pd
from aftermath.core import initial_state
from aftermath.controller import RunContext, episode

SCENARIOS = ('retry_storm', 'db_saturation', 'auth_and_queue', 'healthy', 'mixed_failure')
SEEDS = range(200, 212)  # Reserved seeds not used in the Streamlit benchmark (100–111)
INTENSITY = 1.5
ROWS = []

print('AFTERMATH | HELD-OUT HIDDEN-SEED EVALUATION', flush=True)
print(f'5 synthetic scenarios | seeds 200–211 | uncertainty {INTENSITY} | 120 episodes', flush=True)

for scenario in SCENARIOS:
    for seed in SEEDS:
        for strategy in ('reactive', 'counterfactual'):
            result = episode(
                initial_state(scenario),
                RunContext(seed=seed, uncertainty=INTENSITY),
                strategy,
                max_steps=6,
            )
            ROWS.append({
                key: result[key]
                for key in ('scenario', 'strategy', 'recovered', 'steps',
                            'final_loss', 'peak_loss', 'cumulative_loss',
                            'prediction_mae', 'overrides')
            } | {'seed': seed, 'uncertainty': INTENSITY})

results = pd.DataFrame(ROWS)
filename = 'aftermath_heldout_seeds_200_211.csv'
results.to_csv(filename, index=False)
summary = results.groupby('strategy').agg(
    runs=('recovered', 'size'),
    successful_recoveries=('recovered', 'sum'),
    recovery_rate=('recovered', 'mean'),
    avg_steps=('steps', 'mean'),
    avg_cumulative_loss=('cumulative_loss', 'mean'),
    avg_peak_loss=('peak_loss', 'mean'),
    prediction_mae=('prediction_mae', 'mean'),
)
summary['recovery_rate'] *= 100
print('\nHELD-OUT RESULTS (recovery_rate in %):', flush=True)
print(summary.round(2).to_string(), flush=True)
print(f'\nCSV saved: {filename}', flush=True)
print('Note: new hidden seeds, same five synthetic incident types.', flush=True)
