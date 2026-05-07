import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Reservation Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #f8fafc; }

    .hero-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 50%, #0ea5e9 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(37,99,235,0.2);
    }
    .hero-card h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.4rem 0; }
    .hero-card p  { font-size: 1rem; opacity: 0.88; margin: 0; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
    .metric-card .label { font-size: 0.78rem; color: #64748b; margin-top: 0.3rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }

    .predict-box {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }

    .result-canceled {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #f87171;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-not-canceled {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 2px solid #4ade80;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-sub   { font-size: 0.9rem; opacity: 0.8; }

    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e3a5f;
        border-left: 4px solid #2563eb;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }

    div[data-testid="stSidebar"] { background: #1e3a5f !important; }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] .stNumberInput label { color: #cbd5e1 !important; font-weight: 500; font-size: 0.85rem; }
    div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 { color: white !important; }

    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #0ea5e9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
        box-shadow: 0 4px 12px rgba(37,99,235,0.3);
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(37,99,235,0.4); }

    .insight-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)


# ── Load assets ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_scaler():
    model  = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned.csv", index_col=0)
    df['date of arrival'] = pd.to_datetime(df['date of arrival'], errors='coerce')
    return df

model, scaler = load_model_scaler()
df = load_data()

# Feature lists
FEATURES = [
    'no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights',
    'type_of_meal_plan','required_car_parking_space','room_type_reserved',
    'lead_time','market_segment_type','repeated_guest',
    'no_of_previous_cancellations','no_of_previous_bookings_not_canceled',
    'avg_price_per_room','no_of_special_requests'
]
SCALE_COLS = ['lead_time', 'avg_price_per_room']
CAT_COLS   = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']

MEAL_PLANS   = ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']
ROOM_TYPES   = [f'Room_Type {i}' for i in range(1, 8)]
MARKET_SEGS  = ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary']

# ── Encoding helper ───────────────────────────────────────────────────────────
def encode_and_predict(input_dict):
    row = pd.DataFrame([input_dict])

    # Label encode categoricals using training-set order
    enc_map = {
        'type_of_meal_plan':   {v: i for i, v in enumerate(MEAL_PLANS)},
        'room_type_reserved':  {v: i for i, v in enumerate(ROOM_TYPES)},
        'market_segment_type': {v: i for i, v in enumerate(MARKET_SEGS)},
    }
    for col, mapping in enc_map.items():
        row[col] = row[col].map(mapping)

        # Fail fast if user selected an unexpected category
        if row[col].isna().any():
            valid_vals = list(mapping.keys())
            invalid_val = input_dict.get(col)
            raise ValueError(
                f"Invalid value for '{col}': {invalid_val!r}. "
                f"Expected one of: {valid_vals}"
            )

    # Ensure numeric types are valid before scaling
    row[SCALE_COLS] = row[SCALE_COLS].astype(float)

    # Scale numeric columns
    row[SCALE_COLS] = scaler.transform(row[SCALE_COLS])

    row = row[FEATURES]
    pred  = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    return pred, proba



# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – input form
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏨 Booking Details")
    st.markdown("---")

    st.markdown("### 👥 Guest Info")
    no_of_adults   = st.selectbox("Number of Adults",   [1, 2, 3, 4], index=1)
    no_of_children = st.selectbox("Number of Children", [0, 1, 2, 3], index=0)
    repeated_guest = st.selectbox("Repeated Guest?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    st.markdown("### 🛏️ Stay Details")
    no_of_weekend_nights = st.selectbox("Weekend Nights", [0, 1, 2], index=1)
    no_of_week_nights    = st.selectbox("Week Nights",    [1, 2, 3, 4, 5], index=1)
    room_type = st.selectbox("Room Type", ROOM_TYPES)
    meal_plan = st.selectbox("Meal Plan", MEAL_PLANS)

    st.markdown("### 📋 Booking Info")
    lead_time           = st.slider("Lead Time (days)", 0, 400, 30)
    avg_price_per_room  = st.number_input("Avg Price / Room (€)", 0.0, 600.0, 100.0, step=5.0)
    market_seg          = st.selectbox("Market Segment", MARKET_SEGS)
    car_parking         = st.selectbox("Car Parking Required?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    no_of_special_req   = st.selectbox("Special Requests", [0, 1, 2, 3, 4, 5])

    st.markdown("### 📊 History")
    prev_cancellations  = st.selectbox("Previous Cancellations", list(range(0, 14)), index=0)
    prev_not_cancelled  = st.selectbox("Prev Bookings Not Cancelled", list(range(0, 20)), index=0)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Cancellation")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-card">
  <h1>🏨 Hotel Reservation Cancellation Predictor</h1>
  <p>ML-powered dashboard · Random Forest Classifier · Enter booking details in the sidebar to predict cancellation risk</p>
</div>
""", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔮 Prediction", "📈 Analytics"])

with tab1:
    total        = len(df)
    canceled     = (df['booking_status'] == 'Canceled').sum()
    not_canceled = total - canceled
    cancel_rate  = canceled / total * 100
    avg_price    = df['avg_price_per_room'].mean()
    avg_lead     = df['lead_time'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [f"{total:,}", f"{not_canceled:,}", f"{canceled:,}", f"{cancel_rate:.1f}%", f"€{avg_price:.0f}"],
        ["Total Bookings", "Confirmed", "Cancelled", "Cancellation Rate", "Avg Room Price"]
    ):
        col.markdown(f"""
        <div class="metric-card">
          <div class="value">{val}</div>
          <div class="label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    row1c1, row1c2 = st.columns(2)

    with row1c1:
        st.markdown('<div class="section-header">Booking Status Distribution</div>', unsafe_allow_html=True)
        fig_pie = px.pie(
            values=[not_canceled, canceled],
            names=['Not Canceled', 'Canceled'],
            color_discrete_sequence=['#2563eb', '#f87171'],
            hole=0.45,
        )
        fig_pie.update_traces(textinfo='percent+label', pull=[0, 0.05])
        fig_pie.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with row1c2:
        st.markdown('<div class="section-header">Bookings by Market Segment</div>', unsafe_allow_html=True)
        seg_data = df.groupby(['market_segment_type','booking_status']).size().reset_index(name='count')
        fig_bar = px.bar(
            seg_data, x='market_segment_type', y='count',
            color='booking_status',
            color_discrete_map={'Not_Canceled': '#2563eb', 'Canceled': '#f87171'},
            barmode='group',
        )
        fig_bar.update_layout(
            xaxis_title="Market Segment", yaxis_title="Bookings",
            legend_title="Status",
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=320,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    row2c1, row2c2 = st.columns(2)

    with row2c1:
        st.markdown('<div class="section-header">Lead Time Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            df, x='lead_time', color='booking_status', nbins=50,
            color_discrete_map={'Not_Canceled': '#2563eb', 'Canceled': '#f87171'},
            opacity=0.75,
        )
        fig_hist.update_layout(
            xaxis_title="Lead Time (days)", yaxis_title="Count",
            barmode='overlay',
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with row2c2:
        st.markdown('<div class="section-header">Room Price vs Cancellation</div>', unsafe_allow_html=True)
        price_status = df.groupby('booking_status')['avg_price_per_room'].mean().reset_index()
        fig_price = px.bar(
            price_status, x='booking_status', y='avg_price_per_room',
            color='booking_status',
            color_discrete_map={'Not_Canceled': '#2563eb', 'Canceled': '#f87171'},
            text_auto='.1f',
        )
        fig_price.update_layout(
            xaxis_title="Booking Status", yaxis_title="Avg Price (€)",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # Meal plan breakdown
    st.markdown('<div class="section-header">Cancellation Rate by Meal Plan & Room Type</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    with rc1:
        meal_cancel = df.groupby('type_of_meal_plan').apply(
            lambda x: (x['booking_status']=='Canceled').mean()*100
        ).reset_index(name='cancel_rate')
        fig_meal = px.bar(meal_cancel, x='type_of_meal_plan', y='cancel_rate',
                          color='cancel_rate', color_continuous_scale='RdYlGn_r',
                          text_auto='.1f')
        fig_meal.update_layout(xaxis_title="Meal Plan", yaxis_title="Cancellation Rate (%)",
                               coloraxis_showscale=False,
                               margin=dict(t=10,b=10,l=10,r=10),
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280)
        st.plotly_chart(fig_meal, use_container_width=True)
    with rc2:
        room_cancel = df.groupby('room_type_reserved').apply(
            lambda x: (x['booking_status']=='Canceled').mean()*100
        ).reset_index(name='cancel_rate')
        fig_room = px.bar(room_cancel, x='room_type_reserved', y='cancel_rate',
                          color='cancel_rate', color_continuous_scale='RdYlGn_r',
                          text_auto='.1f')
        fig_room.update_layout(xaxis_title="Room Type", yaxis_title="Cancellation Rate (%)",
                               coloraxis_showscale=False,
                               margin=dict(t=10,b=10,l=10,r=10),
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280)
        st.plotly_chart(fig_room, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Fill in the sidebar fields and click **Predict Cancellation**")

    if predict_btn:
        input_data = {
            'no_of_adults': no_of_adults,
            'no_of_children': no_of_children,
            'no_of_weekend_nights': no_of_weekend_nights,
            'no_of_week_nights': no_of_week_nights,
            'type_of_meal_plan': meal_plan,
            'required_car_parking_space': car_parking,
            'room_type_reserved': room_type,
            'lead_time': lead_time,
            'market_segment_type': market_seg,
            'repeated_guest': repeated_guest,
            'no_of_previous_cancellations': prev_cancellations,
            'no_of_previous_bookings_not_canceled': prev_not_cancelled,
            'avg_price_per_room': avg_price_per_room,
            'no_of_special_requests': no_of_special_req,
        }

        pred, proba = encode_and_predict(input_data)

        # Map probabilities to class labels robustly (do not assume proba[0] == Canceled)
        classes = list(model.classes_)
        # Dataset labels in this project are: 0 => Canceled, 1 => Not_Canceled (but keep it robust)
        cancel_class = 0
        not_cancel_class = 1

        if cancel_class in classes and not_cancel_class in classes:
            cancel_idx = classes.index(cancel_class)
            not_cancel_idx = classes.index(not_cancel_class)
            cancel_prob = float(proba[cancel_idx]) * 100
            no_cancel_prob = float(proba[not_cancel_idx]) * 100
        else:
            # Fallback to old behavior
            cancel_prob = float(proba[0]) * 100
            no_cancel_prob = float(proba[1]) * 100


        pc1, pc2 = st.columns([1, 1])

        with pc1:
            if pred == 0:  # Canceled
                st.markdown(f"""
                <div class="result-canceled">
                  <div class="result-title">⚠️ Likely to Cancel</div>
                  <div class="result-sub">Model predicts this booking will be cancelled</div>
                  <br>
                  <div style="font-size:2.5rem;font-weight:800;color:#dc2626;">{cancel_prob:.1f}%</div>
                  <div style="font-size:0.85rem;color:#7f1d1d;">Cancellation Probability</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-not-canceled">
                  <div class="result-title">✅ Likely to Stay</div>
                  <div class="result-sub">Model predicts this booking will be honored</div>
                  <br>
                  <div style="font-size:2.5rem;font-weight:800;color:#16a34a;">{no_cancel_prob:.1f}%</div>
                  <div style="font-size:0.85rem;color:#14532d;">Confirmation Probability</div>
                </div>""", unsafe_allow_html=True)

            # Gauge chart
            st.markdown("")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cancel_prob,
                title={'text': "Cancellation Risk", 'font': {'size': 14}},
                number={'suffix': '%', 'font': {'size': 28}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#f87171" if cancel_prob > 50 else "#4ade80"},
                    'steps': [
                        {'range': [0, 30],  'color': '#dcfce7'},
                        {'range': [30, 60], 'color': '#fef9c3'},
                        {'range': [60, 100],'color': '#fee2e2'},
                    ],
                    'threshold': {'line': {'color': "#1e3a5f", 'width': 3}, 'value': 50},
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20),
                                    paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)

        with pc2:
            st.markdown("#### 📋 Booking Summary")
            summary = {
                "Guests":        f"{no_of_adults} adults, {no_of_children} children",
                "Stay":          f"{no_of_weekend_nights} weekend + {no_of_week_nights} weekday nights",
                "Room":          room_type,
                "Meal Plan":     meal_plan,
                "Lead Time":     f"{lead_time} days",
                "Price/Night":   f"€{avg_price_per_room:.2f}",
                "Market":        market_seg,
                "Parking":       "Yes" if car_parking else "No",
                "Repeated Guest":"Yes" if repeated_guest else "No",
                "Special Req.":  no_of_special_req,
                "Prev Cancels":  prev_cancellations,
            }
            for k, v in summary.items():
                st.markdown(f"**{k}:** {v}")

            st.markdown("#### 💡 Risk Factors")
            risks = []
            if lead_time > 150:   risks.append("Long lead time increases cancellation risk")
            if prev_cancellations > 0: risks.append("Guest has previous cancellations")
            if no_of_special_req == 0: risks.append("No special requests — lower commitment signal")
            if market_seg == "Online": risks.append("Online bookings have higher cancellation rates")
            if not risks:
                risks.append("No major risk factors detected ✅")

            for r in risks:
                st.markdown(f'<div class="insight-box">• {r}</div>', unsafe_allow_html=True)

    else:
        st.info("👈 Configure booking details in the sidebar and click **Predict Cancellation** to get results.")

        # Show feature importance from model
        st.markdown('<div class="section-header">Model Feature Importance</div>', unsafe_allow_html=True)
        feat_imp = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig_imp = px.bar(
            feat_imp, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Blues',
        )
        fig_imp.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=420,
            yaxis_title="",
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Cancellation Rate by Lead Time Bucket</div>', unsafe_allow_html=True)
    df_a = df.copy()
    df_a['lead_bucket'] = pd.cut(df_a['lead_time'],
                                  bins=[0,30,60,100,150,200,300,500],
                                  labels=['0-30','31-60','61-100','101-150','151-200','201-300','300+'])
    lt_cancel = df_a.groupby('lead_bucket', observed=True).apply(
        lambda x: (x['booking_status']=='Canceled').mean()*100
    ).reset_index(name='cancel_rate')
    fig_lt = px.line(lt_cancel, x='lead_bucket', y='cancel_rate', markers=True,
                     color_discrete_sequence=['#2563eb'])
    fig_lt.update_layout(xaxis_title="Lead Time Bucket (days)", yaxis_title="Cancellation Rate (%)",
                         margin=dict(t=10,b=10,l=10,r=10),
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
    st.plotly_chart(fig_lt, use_container_width=True)

    ac1, ac2 = st.columns(2)

    with ac1:
        st.markdown('<div class="section-header">Special Requests vs Cancellation</div>', unsafe_allow_html=True)
        sr_data = df.groupby('no_of_special_requests').apply(
            lambda x: (x['booking_status']=='Canceled').mean()*100
        ).reset_index(name='cancel_rate')
        fig_sr = px.bar(sr_data, x='no_of_special_requests', y='cancel_rate',
                        color='cancel_rate', color_continuous_scale='RdYlGn_r', text_auto='.1f')
        fig_sr.update_layout(xaxis_title="# Special Requests", yaxis_title="Cancellation Rate (%)",
                              coloraxis_showscale=False,
                              margin=dict(t=10,b=10,l=10,r=10),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280)
        st.plotly_chart(fig_sr, use_container_width=True)

    with ac2:
        st.markdown('<div class="section-header">Repeated Guest vs Cancellation</div>', unsafe_allow_html=True)
        rg_data = df.groupby('repeated_guest').apply(
            lambda x: (x['booking_status']=='Canceled').mean()*100
        ).reset_index(name='cancel_rate')
        rg_data['repeated_guest'] = rg_data['repeated_guest'].map({0:'New Guest', 1:'Repeated Guest'})
        fig_rg = px.bar(rg_data, x='repeated_guest', y='cancel_rate',
                        color='repeated_guest',
                        color_discrete_sequence=['#f87171','#2563eb'], text_auto='.1f')
        fig_rg.update_layout(xaxis_title="Guest Type", yaxis_title="Cancellation Rate (%)",
                              showlegend=False,
                              margin=dict(t=10,b=10,l=10,r=10),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280)
        st.plotly_chart(fig_rg, use_container_width=True)

    # Monthly trends
    st.markdown('<div class="section-header">Monthly Arrival Trends</div>', unsafe_allow_html=True)
    df_m = df.copy()
    df_m['month'] = df_m['date of arrival'].dt.month
    monthly = df_m.groupby(['month','booking_status']).size().reset_index(name='count')
    fig_month = px.bar(monthly, x='month', y='count', color='booking_status', barmode='group',
                       color_discrete_map={'Not_Canceled':'#2563eb','Canceled':'#f87171'},
                       labels={'month':'Month','count':'Bookings'})
    fig_month.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(1,13)),
                                       ticktext=['Jan','Feb','Mar','Apr','May','Jun',
                                                 'Jul','Aug','Sep','Oct','Nov','Dec']),
                             margin=dict(t=10,b=10,l=10,r=10),
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=320)
    st.plotly_chart(fig_month, use_container_width=True)

    # Raw data
    with st.expander("🔍 Explore Raw Data"):
        col_filter = st.selectbox("Filter by Status", ["All", "Canceled", "Not_Canceled"])
        display_df = df if col_filter == "All" else df[df['booking_status'] == col_filter]
        st.dataframe(display_df.head(500), use_container_width=True)
        st.caption(f"Showing up to 500 rows of {len(display_df):,} total")
