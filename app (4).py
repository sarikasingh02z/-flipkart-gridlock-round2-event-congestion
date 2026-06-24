import gradio as gr
import pandas as pd
import pickle

with open('model_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

xgb_model   = artifacts['xgb_model']
le_cause    = artifacts['le_cause']
le_corridor = artifacts['le_corridor']
le_day      = artifacts['le_day']
le_priority = artifacts['le_priority']
le_closure  = artifacts['le_closure']
le_target   = artifacts['le_target']

PEAK_HOURS = [19, 20, 21]

DIVERSION_MAP = {
    'Mysore Road'      : ['NICE Road', 'Kanakapura Road', 'Bannerghatta Road'],
    'Bellary Road 1'   : ['Hebbal Flyover alternate', 'Outer Ring Road North'],
    'Bellary Road 2'   : ['Ballari Road service lane', 'Mehkri Circle bypass'],
    'Tumkur Road'      : ['Yeshwanthpur bypass', 'Peenya Industrial route'],
    'Hosur Road'       : ['Sarjapur Road', 'Bannerghatta Road'],
    'ORR North 1'      : ['Hennur Road', 'Thanisandra Main Road'],
    'ORR North 2'      : ['Outer Ring Road alternate sectors'],
    'Old Madras Road'  : ['KR Pura bypass', 'Whitefield Road'],
    'ORR East 1'       : ['Marathahalli bridge alternate', 'Sarjapur Road'],
    'Bannerghata Road' : ['JP Nagar ring road', 'Kanakapura Road'],
    'Magadi Road'      : ['Chord Road', 'Rajajinagar alternate'],
    'ORR West 1'       : ['Mysore Road link', 'NICE Road connector'],
}

RECS = {
    'None'   : {'officers': 0,  'barricading': False, 'urgency': 'No action', 'patrol': 'Standard patrol only'},
    'Low'    : {'officers': 3,  'barricading': False, 'urgency': 'Monitor',   'patrol': 'Deploy 1 mobile unit'},
    'Medium' : {'officers': 7,  'barricading': True,  'urgency': 'Prepare',   'patrol': 'Deploy 2 units + 1 traffic warden'},
    'High'   : {'officers': 15, 'barricading': True,  'urgency': 'URGENT',    'patrol': 'Deploy QRT + traffic sub-inspector + divert'},
}

def predict_congestion(event_cause, corridor, hour, day_of_week, priority, requires_closure):
    try:
        cause_e = le_cause.transform([event_cause])[0]
    except:
        cause_e = 0
    try:
        corridor_e = le_corridor.transform([corridor])[0]
    except:
        corridor_e = 0
    try:
        day_e = le_day.transform([day_of_week])[0]
    except:
        day_e = 0

    priority_e = le_priority.transform([priority])[0]
    closure_e  = le_closure.transform([requires_closure])[0]
    is_peak    = int(hour in PEAK_HOURS)
    is_wknd    = int(day_of_week in ['Saturday', 'Sunday'])

    X_new = pd.DataFrame([{
        'cause_enc'   : cause_e,
        'corridor_enc': corridor_e,
        'hour'        : hour,
        'day_enc'     : day_e,
        'priority_enc': priority_e,
        'closure_enc' : closure_e,
        'is_peak_hour': is_peak,
        'is_weekend'  : is_wknd,
        'duration_hrs': 2.0
    }])

    proba_encoded = xgb_model.predict_proba(X_new)[0]
    cat_encoded   = xgb_model.predict(X_new)[0]
    cat           = le_target.inverse_transform([cat_encoded])[0]

    # Safety floor
    if priority == 'High' and cat == 'None':
        cat = 'Low'

    rec  = RECS[cat]
    divs = DIVERSION_MAP.get(corridor, ['Use alternate city routes'])

    output = f"""
EVENT IMPACT FORECAST
======================
Event       : {event_cause.replace('_',' ').title()}
Corridor    : {corridor}
Time        : {hour:02d}:00 on {day_of_week}
Road Closure: {requires_closure}

Predicted Impact : {cat.upper()}
Urgency          : {rec['urgency']}

Officers Required : {rec['officers']}                                                                                                            
Barricading       : {"YES" if rec['barricading'] else "NO"}
Patrol Action     : {rec['patrol']}

Suggested Diversions:
{chr(10).join(['  → ' + d for d in divs])}

Model Confidence  : {max(proba_encoded):.0%}
"""
    return output

demo = gr.Interface(
    fn=predict_congestion,
    inputs=[
        gr.Dropdown(['procession', 'construction', 'public_event', 'accident', 'vehicle_breakdown'], label="Event Type", value='procession'),
        gr.Dropdown(list(DIVERSION_MAP.keys()), label="Corridor", value='Mysore Road'),
        gr.Slider(0, 23, step=1, value=12, label="Hour"),
        gr.Dropdown(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], label="Day", value='Friday'),
        gr.Dropdown(['High', 'Low'], label="Priority", value='High'),
        gr.Radio(['TRUE', 'FALSE'], label="Requires Road Closure", value='TRUE')
    ],
    outputs=gr.Textbox(label="Congestion Forecast & Recommendations", lines=15),
    title="Bengaluru Event-Driven Congestion Intelligence",
    description="Forecast event-driven traffic disruptions and deliver actionable deployment strategies for minimizing citywide impact."

    
)

demo.launch()