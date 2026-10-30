"""Interactive incident recovery experiments and model-assisted investigation."""
import os
from dataclasses import asdict
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from aftermath.core import initial_state, observations, loss, recovered, ACTIONS, SCENARIOS
from aftermath.planner import plan
from aftermath.ml import assess, predict_action_loss, impact_model_summary
from aftermath.controller import RunContext, step, episode, VALID_STRATEGIES

st.set_page_config(page_title='AFTERMATH · AI Recovery Lab',page_icon='🪐',layout='wide')

st.markdown("""
<style>
/* Main page background */
.stApp {
    background-color: #080B09;
    color: #E8F5EC;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #111A15;
}

/* Main headings */
h1, h2, h3 {
    color: #20F28B !important;
    letter-spacing: -0.025em;
}

/* Large metric cards */
[data-testid="stMetric"] {
    background-color: #142019;
    border: 1px solid #28513B;
    border-radius: 10px;
    padding: 12px;
}

/* Keep original page spacing */
.block-container {
    padding-top: 1.3rem;
    max-width: 1250px;
}
</style>
""", unsafe_allow_html=True)


st.title('AFTERMATH')
st.caption('AI investigator + counterfactual safety gate · Recovery under uncertainty')
st.info('project sandbox synthetic telemetry and recovery actions only no real production services are connected.', icon='ℹ️')

with st.sidebar:
    st.header('Experiment controls')
    scenario = st.selectbox('Incident scenario',SCENARIOS)
    mode = st.selectbox('Recovery strategy', ['counterfactual','reactive','llm_only','llm_guarded'], format_func=lambda x: {'reactive':'Reactive heuristic (no AI)','counterfactual':'Counterfactual planner (no LLM)','llm_only':'Live LLM investigator · no gate','llm_guarded':'Live LLM investigator + safety gate'}[x])
    seed = st.number_input('Hidden-environment seed',min_value=0,max_value=100000,value=42,step=1)
    intensity = st.slider('Prediction mismatch intensity',0.0,2.0,1.0,0.1,help='0 means the nominal digital twin is exact; higher settings increase hidden differences.')
    st.divider()
    st.subheader('Model connection')
    api_key = st.text_input('API key',type='password',value='')
    model = st.text_input('Model ID',value=os.getenv('OPENAI_MODEL',''),placeholder='Provider-specific Qwen model ID')
    base_url = st.text_input('OpenAI-compatible API base URL',value=os.getenv('OPENAI_BASE_URL',''),placeholder='Provider API endpoint ending in /v1')
    
    if api_key: st.success('API key provided. Connection will be checked on the first model call.')
    else: st.caption('Local planning and fitted ML models work without a hosted-model connection.')

signature = (scenario,int(seed),float(intensity))
if st.session_state.get('signature') != signature:
    st.session_state.signature = signature
    st.session_state.state = initial_state(scenario)
    st.session_state.context = RunContext(seed=int(seed),uncertainty=float(intensity))
    st.session_state.events = []
    st.session_state.comparison = None
    st.session_state.ai_comparison = None

s = st.session_state.state
context = st.session_state.context
obs = observations(s)
score = assess(s)

summary, live, evaluation, technical = st.tabs(['System & recovery','Agent investigation','Experiment lab','Architecture & limitations'])
with summary:
    a,b,c,d,e = st.columns(5)
    a.metric('System loss · synthetic',f'{loss(s):.1f}')
    b.metric('Error rate',f"{obs['metrics']['error_rate_pct']}%")
    c.metric('DB utilization',f"{obs['metrics']['db_util_pct']}%")
    d.metric('Recovered?','YES' if recovered(s) else 'NO')
    e.metric('ML anomaly?', 'YES' if score['anomaly'] else 'NO')
    left,right=st.columns([0.85,1.15])
    with left:
        st.subheader('Observed telemetry')
        st.json(obs,expanded=False)
        st.caption(f"IsolationForest synthetic anomaly score: {score['anomaly_score']}. A negative decision score is anomalous; this is not a probability.")
        st.code('Authentication → Payments → Database\n                     ↘ Notifications',language=None)
        with st.expander('Real trained ML: predicted loss if each action is executed'):
            st.dataframe(pd.DataFrame([{'action':a, 'RF predicted next loss':predict_action_loss(s,a)} for a in ACTIONS]),hide_index=True,use_container_width=True)
            st.caption(f"RandomForest holdout MAE on synthetic nominal transitions: {impact_model_summary()['synthetic_holdout_MAE']} synthetic loss points. This does not validate production predictions.")
        
    with right:
        st.subheader('Alternative futures · nominal digital twin')
        options = plan(s,top_k=5)
        if options[0]['actions']:
            fig=go.Figure()
            for i,o in enumerate(options[:4]):
                fig.add_trace(go.Scatter(x=list(range(len(o['losses']))),y=o['losses'],mode='lines+markers',name=f"Future {i+1}"))
            fig.update_layout(height=290,xaxis_title='Planned action step',yaxis_title='Synthetic loss',legend=dict(orientation='h',y=-.25),margin=dict(t=15,b=45))
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(pd.DataFrame([{'rank':i+1,'recovery_sequence':' → '.join(o['actions']), 'risk_score':o['score'],'predicted_recovery':o['recovered']} for i,o in enumerate(options)]),hide_index=True,use_container_width=True)
        else:
            st.success('System has already recovered.')

    st.subheader('Control the recovery')
    st.caption('Live LLM modes invoke a real hosted model and may charge for multiple tool calls. The model only proposes actions; actual execution remains within the simulator.')
    c1,c2,c3=st.columns(3)
    def apply_run(n):
        if mode.startswith('llm_') and not api_key:
            st.error('A hosted-model API key is required for LLM strategies.')
            return
        if mode.startswith('llm_') and (not model.strip() or not base_url.strip()):
            st.error('Set the model ID and compatible API base URL before running the investigator.')
            return
        try:
            with st.spinner('Investigating incident, evaluating actions, executing in the simulator…'):
                for _ in range(n):
                    if recovered(st.session_state.state): break
                    next_s, ev = step(st.session_state.state, st.session_state.context,mode,
                        api_key=api_key,model=model,base_url=base_url)
                    st.session_state.state=next_s
                    st.session_state.events.append(ev)
            st.rerun()
        except Exception as exc:
            st.error(f'{type(exc).__name__}: {str(exc)[:900]}')
            st.info('Verify the model ID, API endpoint and function-calling support.')
    if c1.button('Execute ONE action',type='primary',disabled=recovered(s),use_container_width=True): apply_run(1)
    if c2.button('Auto recover · max 6',disabled=recovered(s),use_container_width=True): apply_run(6)
    if c3.button('Reset incident',use_container_width=True):
        st.session_state.state=initial_state(scenario)
        st.session_state.context=RunContext(seed=int(seed),uncertainty=float(intensity))
        st.session_state.events=[]
        st.rerun()
    st.subheader('Executed interventions')
    if st.session_state.events:
        rows=[]
        for ev in st.session_state.events:
            rows.append({'step':ev['step'],'strategy':ev['strategy'],'AI proposed':ev['proposed'],
                         'executed':ev['action'],'gate':ev['gate']['decision'],
                         'loss before':ev['loss_before'], 'predicted loss':ev['predicted_loss'],
                         'actual loss':ev['actual_loss'],'prediction error':ev['prediction_error']})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.download_button('Download recovery trace (CSV)',pd.DataFrame(rows).to_csv(index=False),file_name='aftermath_trace.csv')
        st.caption('Prediction error = observed loss minus predicted loss. Positive values mean the outcome was worse than expected. Hidden conditions are not shown to the investigator.')
    else: st.caption('No actions executed yet.')

with live:
    st.subheader('Model investigation trace')
    if not st.session_state.events: st.info('Traces appear after running an LLM strategy. Heuristic and planning strategies do not call a language model.')
    else:
        ai_events=[ev for ev in st.session_state.events if ev.get('ai')]
        if not ai_events:
            st.info('This run used a heuristic or planner; no hosted language model was called.')
        for ev in reversed(ai_events):
            ai=ev['ai']
            with st.expander(f"Step {ev['step']} · {ai['model']} · proposed {ev['proposed']} → executed {ev['action']}",expanded=True):
                st.write('**Hypothesis:**',ai['hypothesis'])
                st.write('**Evidence:**',ai['evidence'])
                st.write('**Model reason:**',ai['reason'])
                st.write('**Gate verdict:**',ev['gate']['decision'],ev['gate']['reason'])
                st.write('**Token usage (if available):**',ai.get('usage'))
                st.write('**Actual model tool-call trace**')
                for t in ai['trace']:
                    st.markdown(f"**Tool:** `{t['tool']}`")
                    st.json({'arguments':t['arguments'],'result':t['result']},expanded=False)


with evaluation:
    st.subheader('1 · Synthetic uncertainty benchmark')
    st.write('Compares the old reactive heuristic against the nominal counterfactual planner across **independent hidden conditions**. The planning model does not know the actual execution coefficients.')
    seeds = st.slider('Number of hidden-environment seeds per scenario',2,12,5)
    if st.button('Run offline uncertainty benchmark',type='primary'):
        rows=[]
        with st.spinner('Running seeded synthetic experiments…'):
            for incident in SCENARIOS:
                for index in range(seeds):
                    for strategy in ('reactive','counterfactual'):
                        result=episode(initial_state(incident),RunContext(seed=100+index,uncertainty=float(intensity)),strategy,max_steps=6)
                        rows.append({k:result[k] for k in ['scenario','strategy','recovered','steps','final_loss','peak_loss','cumulative_loss','prediction_mae','overrides']} | {'seed':100+index})
        st.session_state.comparison=pd.DataFrame(rows)
    if isinstance(st.session_state.get('comparison'),pd.DataFrame):
        df=st.session_state.comparison
        st.dataframe(df,hide_index=True,use_container_width=True)
        st.download_button('Download offline benchmark CSV',df.to_csv(index=False),file_name='aftermath_uncertain_benchmark.csv')
        st.markdown('**Average outcomes by strategy**')
        agg=df.groupby('strategy',as_index=False).agg(recovery_rate=('recovered','mean'),avg_steps=('steps','mean'),avg_peak_loss=('peak_loss','mean'),avg_cumulative_loss=('cumulative_loss','mean'),prediction_mae=('prediction_mae','mean'))
        st.dataframe(agg,hide_index=True,use_container_width=True)
        fig=go.Figure()
        for name in agg['strategy']:
            part=df[df['strategy']==name]
            fig.add_trace(go.Box(x=part['strategy'],y=part['cumulative_loss'],name=name,boxpoints='all',jitter=.2))
        fig.update_layout(height=330,yaxis_title='Cumulative simulated loss',showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    st.divider()
    st.subheader('2 · LLM-only vs LLM + safety gate')
    st.caption('Runs both strategies separately for a small case. They may produce different LLM proposals; results are not a controlled paired-model benchmark. Repeated evaluations should cache matching proposals.')
    incident=st.selectbox('AI comparison incident',['retry_storm','mixed_failure','db_saturation','auth_and_queue'],key='lab_incident')
    steps=st.slider('Max AI recovery steps per strategy',1,4,2)
    if st.button('Run LLM comparison (API calls)',disabled=not bool(api_key)):
        rows=[]
        with st.spinner('Calling the configured model. This can take multiple API requests…'):
            for strategy in ('llm_only','llm_guarded'):
                try:
                    result=episode(initial_state(incident),RunContext(seed=int(seed),uncertainty=float(intensity)),strategy,max_steps=steps,
                                   api_key=api_key,model=model,base_url=base_url)
                    rows.append({k:result[k] for k in ['scenario','strategy','recovered','steps','final_loss','peak_loss','cumulative_loss','prediction_mae','overrides']})
                except Exception as exc:
                    st.error(f'{strategy}: {type(exc).__name__}: {str(exc)[:400]}')
        if rows: st.session_state.ai_comparison=pd.DataFrame(rows)
    if isinstance(st.session_state.get('ai_comparison'),pd.DataFrame):
        st.dataframe(st.session_state.ai_comparison,hide_index=True,use_container_width=True)
        st.download_button('Download AI comparison CSV',st.session_state.ai_comparison.to_csv(index=False),file_name='aftermath_ai_comparison.csv')

with technical:
    st.subheader('System Architecture')

    st.markdown("""
    **Architecture**

    Simulator → AI Investigator → Safety Gate → Recovery

    **Technologies**

    Python, Streamlit, Qwen, Isolation Forest, Random Forest

    **Limitations**

    - Synthetic environment only.
    - No real production infrastructure integration.
    - Recovery performance has not been validated on real incidents.
    """)
    
    st.markdown('**Technical files**: `aftermath/agent.py` (live LLM), `ml.py` (trained detector), `uncertainty.py` (hidden dynamics), `controller.py` (safety gate), `core.py` (simulator), `planner.py` (counterfactual search), `app.py` (dashboard).')
    
