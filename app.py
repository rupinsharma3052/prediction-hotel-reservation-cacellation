import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Reservation Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9ff, #e8edff);
        border-left: 4px solid #1a73e8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 0.78rem; color: #555; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.6rem; font-weight: 800; color: #1a73e8; }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a1a2e;
        border-bottom: 2px solid #e0e7ff;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem;
    }
    .prediction-box-canceled {
        background: linear-gradient(135deg, #fff0f0, #ffe0e0);
        border: 2px solid #e53935;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .prediction-box-not-canceled {
        background: linear-gradient(135deg, #f0fff4, #d4edda);
        border: 2px solid #43a047;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading & Preprocessing ─────────────────────────────────────────────
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    if "date of arrival" in df.columns:
        df["date of arrival"] = pd.to_datetime(df["date of arrival"], errors="coerce")
    return df

@st.cache_data
def preprocess_and_train(df):
    data = df.copy()
    if "date of arrival" in data.columns:
        data.drop(columns=["date of arrival"], inplace=True)

    le = LabelEncoder()
    cat_cols = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    data["booking_status"] = data["booking_status"].map({"Canceled": 1, "Not_Canceled": 0})

    X = data.drop(columns=["booking_status"])
    y = data["booking_status"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=11,
                                                min_samples_leaf=1, min_samples_split=6,
                                                random_state=0),
        "Random Forest": RandomForestClassifier(bootstrap=False, criterion="gini",
                                                max_depth=None, min_samples_leaf=2,
                                                min_samples_split=2, n_estimators=10,
                                                random_state=0),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "pred": pred,
            "accuracy": accuracy_score(y_test, pred),
            "mae": mean_absolute_error(y_test, pred),
            "mse": mean_squared_error(y_test, pred),
            "cm": confusion_matrix(y_test, pred),
            "report": classification_report(y_test, pred, output_dict=True),
        }

    feature_names = X.columns.tolist()
    fi = pd.DataFrame({
        "Feature": feature_names,
        "Importance": results["Random Forest"]["model"].feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return results, scaler, X.columns.tolist(), y_test, fi

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏨 Navigation")
    page = st.radio("", [
        "📊 Overview & EDA",
        "🤖 Model Performance",
        "🔮 Predict Cancellation"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 📂 Upload Dataset")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    st.markdown("*Or use the default Hotel Reservations dataset.*")

# ─── Load Data ─────────────────────────────────────────────────────────────────
DEFAULT_PATH = "cleaned.csv"

if uploaded:
    df = load_data(uploaded)
    st.sidebar.success("✅ Custom dataset loaded!")
else:
    try:
        df = load_data(DEFAULT_PATH)
    except Exception:
        st.error("❌ Default dataset not found. Please upload a CSV file.")
        st.stop()

results, scaler, feature_names, y_test, feature_importance = preprocess_and_train(df)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏨 Hotel Reservation Cancellation Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">End-to-end ML dashboard — EDA · Model Evaluation · Live Predictions</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW & EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    canceled = (df["booking_status"] == "Canceled").sum()
    not_canceled = total - canceled
    cancel_rate = canceled / total * 100

    for col, label, value in zip(
        [col1, col2, col3, col4],
        ["Total Bookings", "Cancellations", "Not Canceled", "Cancellation Rate"],
        [f"{total:,}", f"{canceled:,}", f"{not_canceled:,}", f"{cancel_rate:.1f}%"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: Cancellation status + Market Segment
    st.markdown('<div class="section-header">Booking Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        status_counts = df["booking_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(status_counts, names="Status", values="Count",
                     color="Status",
                     color_discrete_map={"Canceled": "#ef5350", "Not_Canceled": "#42a5f5"},
                     title="Booking Status Distribution", hole=0.4)
        fig.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        seg = df.groupby(["market_segment_type", "booking_status"]).size().reset_index(name="count")
        fig = px.bar(seg, x="market_segment_type", y="count", color="booking_status",
                     barmode="group", title="Cancellations by Market Segment",
                     color_discrete_map={"Canceled": "#ef5350", "Not_Canceled": "#42a5f5"},
                     labels={"market_segment_type": "Market Segment", "count": "Count", "booking_status": "Status"})
        fig.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Lead time + Avg price
    st.markdown('<div class="section-header">Pricing & Lead Time Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(df, x="lead_time", color="booking_status", nbins=60,
                           title="Lead Time Distribution by Status",
                           color_discrete_map={"Canceled": "#ef5350", "Not_Canceled": "#42a5f5"},
                           labels={"lead_time": "Lead Time (days)", "booking_status": "Status"},
                           barmode="overlay", opacity=0.75)
        fig.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(df, x="booking_status", y="avg_price_per_room", color="booking_status",
                     title="Avg Price per Room by Status",
                     color_discrete_map={"Canceled": "#ef5350", "Not_Canceled": "#42a5f5"},
                     labels={"avg_price_per_room": "Avg Price (€)", "booking_status": "Status"})
        fig.update_layout(margin=dict(t=40, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Room type + Special requests
    st.markdown('<div class="section-header">Room & Guests</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        room = df.groupby(["room_type_reserved", "booking_status"]).size().reset_index(name="count")
        fig = px.bar(room, x="room_type_reserved", y="count", color="booking_status",
                     title="Room Type vs Cancellation",
                     color_discrete_map={"Canceled": "#ef5350", "Not_Canceled": "#42a5f5"},
                     labels={"room_type_reserved": "Room Type", "count": "Count", "booking_status": "Status"})
        fig.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sr = df.groupby(["no_of_special_requests", "booking_status"]).size().reset_index(name="count")
        fig = px.bar(sr, x="no_of_special_requests", y="count", color="booking_status",
                     title="Special Requests vs Cancellation",
                     color_discrete_map={"Canceled": "#ef5350", "Not_Canceled": "#42a5f5"},
                     labels={"no_of_special_requests": "Special Requests", "count": "Count", "booking_status": "Status"})
        fig.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Arrival trends (if date column present)
    if "date of arrival" in df.columns and df["date of arrival"].notna().sum() > 0:
        st.markdown('<div class="section-header">Arrival Trends</div>', unsafe_allow_html=True)
        df_date = df.dropna(subset=["date of arrival"]).copy()
        df_date["month"] = df_date["date of arrival"].dt.month
        monthly = df_date.groupby(["month", "booking_status"]).size().reset_index(name="count")
        fig = px.line(monthly, x="month", y="count", color="booking_status",
                      markers=True, title="Monthly Bookings & Cancellations",
                      color_discrete_map={"Canceled": "#ef5350", "Not_Canceled": "#42a5f5"},
                      labels={"month": "Month", "count": "Count", "booking_status": "Status"})
        fig.update_xaxes(tickvals=list(range(1, 13)),
                         ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
        fig.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Raw data preview
    with st.expander("🔍 Preview Raw Dataset"):
        st.dataframe(df.head(100), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

    # Metrics table
    metrics_rows = []
    for name, r in results.items():
        metrics_rows.append({
            "Model": name,
            "Accuracy": f"{r['accuracy']:.4f}",
            "MAE": f"{r['mae']:.4f}",
            "MSE": f"{r['mse']:.4f}",
        })
    metrics_df = pd.DataFrame(metrics_rows)
    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

    # Bar charts
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]
    maes = [results[n]["mae"] for n in names]
    mses = [results[n]["mse"] for n in names]

    c1, c2, c3 = st.columns(3)
    for col, metric_vals, metric_name, color in zip(
        [c1, c2, c3],
        [accuracies, maes, mses],
        ["Accuracy", "MAE", "MSE"],
        ["#42a5f5", "#ffa726", "#ef5350"]
    ):
        fig = px.bar(x=names, y=metric_vals, title=metric_name,
                     labels={"x": "Model", "y": metric_name},
                     color_discrete_sequence=[color])
        fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
        col.plotly_chart(fig, use_container_width=True)

    # Confusion Matrices
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (name, r) in zip(cols, results.items()):
        cm = r["cm"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual"),
                        x=["Not Canceled", "Canceled"],
                        y=["Not Canceled", "Canceled"],
                        title=name)
        fig.update_layout(margin=dict(t=40, b=10))
        col.plotly_chart(fig, use_container_width=True)

    # Classification reports
    st.markdown('<div class="section-header">Classification Reports</div>', unsafe_allow_html=True)
    tabs = st.tabs(list(results.keys()))
    for tab, (name, r) in zip(tabs, results.items()):
        with tab:
            report_df = pd.DataFrame(r["report"]).T.drop(columns=["support"], errors="ignore")
            report_df = report_df.dropna()
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    # Feature Importance
    st.markdown('<div class="section-header">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    fig = px.bar(feature_importance, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues",
                 title="Feature Importance — Random Forest")
    fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PREDICT CANCELLATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Cancellation":
    st.markdown('<div class="section-header">Enter Booking Details</div>', unsafe_allow_html=True)
    st.info("Fill in the booking information below to predict whether the reservation will be **Canceled** or **Not Canceled**.")

    meal_map = {"Meal Plan 1": 0, "Meal Plan 2": 1, "Meal Plan 3": 2, "Not Selected": 3}
    room_map = {f"Room_Type {i}": i - 1 for i in range(1, 8)}
    seg_map = {"Aviation": 0, "Complementary": 1, "Corporate": 2, "Offline": 3, "Online": 4}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👥 Guest Info**")
        no_of_adults = st.slider("Number of Adults", 1, 4, 2)
        no_of_children = st.slider("Number of Children", 0, 3, 0)
        repeated_guest = st.selectbox("Repeated Guest?", ["No", "Yes"])
        repeated_guest_val = 1 if repeated_guest == "Yes" else 0

    with col2:
        st.markdown("**🛏️ Room & Stay**")
        room_type = st.selectbox("Room Type Reserved", list(room_map.keys()))
        no_of_weekend_nights = st.slider("Weekend Nights", 0, 2, 1)
        no_of_week_nights = st.slider("Week Nights", 0, 5, 2)
        meal_plan = st.selectbox("Meal Plan", list(meal_map.keys()))

    with col3:
        st.markdown("**📋 Booking Details**")
        lead_time = st.slider("Lead Time (days)", 0, 400, 30)
        market_segment = st.selectbox("Market Segment", list(seg_map.keys()))
        avg_price = st.number_input("Avg Price per Room (€)", min_value=0.0, max_value=600.0, value=100.0, step=5.0)
        no_of_special_requests = st.slider("Special Requests", 0, 5, 0)

    col4, col5 = st.columns(2)
    with col4:
        required_car_parking = st.selectbox("Parking Required?", ["No", "Yes"])
        parking_val = 1 if required_car_parking == "Yes" else 0
    with col5:
        no_previous_cancellations = st.slider("Previous Cancellations", 0, 13, 0)
        no_previous_not_canceled = st.slider("Previous Bookings Not Canceled", 0, 58, 0)

    # Model selector
    st.markdown("---")
    selected_model = st.selectbox("🤖 Choose Prediction Model", list(results.keys()))

    # Predict button
    if st.button("🔮 Predict Now", type="primary", use_container_width=True):
        input_data = np.array([[
            no_of_adults,
            no_of_children,
            no_of_weekend_nights,
            no_of_week_nights,
            meal_map[meal_plan],
            parking_val,
            room_map[room_type],
            lead_time,
            seg_map[market_segment],
            repeated_guest_val,
            no_previous_cancellations,
            no_previous_not_canceled,
            avg_price,
            no_of_special_requests,
        ]])

        # Ensure the feature count matches training
        expected = len(feature_names)
        current = input_data.shape[1]
        if current < expected:
            input_data = np.hstack([input_data, np.zeros((1, expected - current))])
        elif current > expected:
            input_data = input_data[:, :expected]

        input_scaled = scaler.transform(input_data)
        model = results[selected_model]["model"]
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        if prediction == 1:
            st.markdown("""
            <div class="prediction-box-canceled">
                <h2 style="color:#e53935; margin:0;">⚠️ CANCELLATION LIKELY</h2>
                <p style="color:#b71c1c; font-size:1.1rem; margin-top:0.5rem;">
                    This booking is predicted to be <strong>Canceled</strong>.
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-box-not-canceled">
                <h2 style="color:#43a047; margin:0;">✅ BOOKING CONFIRMED</h2>
                <p style="color:#1b5e20; font-size:1.1rem; margin-top:0.5rem;">
                    This booking is predicted to be <strong>Not Canceled</strong>.
                </p>
            </div>""", unsafe_allow_html=True)

        if proba is not None:
            st.markdown("#### Prediction Confidence")
            c1, c2 = st.columns(2)
            c1.metric("🟢 Not Canceled Probability", f"{proba[0]*100:.1f}%")
            c2.metric("🔴 Canceled Probability", f"{proba[1]*100:.1f}%")

            fig = go.Figure(go.Bar(
                x=["Not Canceled", "Canceled"],
                y=[proba[0] * 100, proba[1] * 100],
                marker_color=["#42a5f5", "#ef5350"],
                text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                textposition="outside",
            ))
            fig.update_layout(
                title="Prediction Probability",
                yaxis=dict(range=[0, 110], title="Probability (%)"),
                xaxis_title="Class",
                margin=dict(t=40, b=10),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Model context
        acc = results[selected_model]["accuracy"]
        st.caption(f"ℹ️ Model used: **{selected_model}** — Test Accuracy: **{acc*100:.2f}%**")