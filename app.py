import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import plotly.express as px

st.set_page_config(page_title="Traffic Accident Risk Predictor", layout="wide", page_icon="🚦")

# --- Load everything ONCE and cache it -- avoids reloading the model on every click ---
@st.cache_resource
def load_assets():
    model = joblib.load("models/xgb_model.joblib")
    with open("models/model_config.json") as f:
        config = json.load(f)
    with open("models/deployment_metadata.json") as f:
        meta = json.load(f)
    return model, config, meta

model, config, meta = load_assets()

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

st.title("🚦 Traffic Accident Severity Risk Predictor")
st.caption(
    "Predicts whether a US traffic accident is likely to be **high-severity** "
    "(injury/fatality-level) based on conditions at the time and place. "
    "Trained on 500k+ records from the US-Accidents dataset (Moosavi et al.)."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Risk Calculator", "🗺️ Accident Heatmap", "🌦️ Weather Influence", "📊 Feature Importance"]
)

# ============================================================
# TAB 1: RISK CALCULATOR
# ============================================================
with tab1:
    st.header("Predict Risk for a Given Scenario")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Location & Time")
        state_list = sorted(meta['state_to_code'].keys())
        state = st.selectbox("State", state_list)
        hour = st.slider("Hour of Day (24h)", 0, 23, 8)
        weekday_name = st.selectbox(
            "Day of Week",
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
        month = st.slider("Month", 1, 12, 6)
        is_day = st.radio("Light Condition", ["Day", "Night"], horizontal=True) == "Day"

    with col2:
        st.subheader("🌦️ Weather")
        weather = st.selectbox("Weather Condition", meta['weather_categories'])
        wind_dir = st.selectbox("Wind Direction", meta['wind_categories'])
        temperature = st.number_input("Temperature (°F)", value=meta['numeric_defaults']['Temperature(F)'])
        humidity = st.number_input("Humidity (%)", value=meta['numeric_defaults']['Humidity(%)'], min_value=0.0, max_value=100.0)

    with col3:
        st.subheader("🛣️ Road Features")
        junction = st.checkbox("Junction")
        traffic_signal = st.checkbox("Traffic Signal")
        crossing = st.checkbox("Crossing")
        stop = st.checkbox("Stop Sign")
        station = st.checkbox("Station nearby")

    with st.expander("⚙️ Advanced settings (weather detail, more road features, data source)"):
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            pressure = st.number_input("Pressure (in)", value=meta['numeric_defaults']['Pressure(in)'])
            visibility = st.number_input("Visibility (mi)", value=meta['numeric_defaults']['Visibility(mi)'])
            wind_speed = st.number_input("Wind Speed (mph)", value=meta['numeric_defaults']['Wind_Speed(mph)'])
            precipitation = st.number_input("Precipitation (in)", value=0.0)
        with adv2:
            amenity = st.checkbox("Amenity")
            bump = st.checkbox("Bump")
            give_way = st.checkbox("Give Way")
            no_exit = st.checkbox("No Exit")
        with adv3:
            railway = st.checkbox("Railway crossing")
            traffic_calming = st.checkbox("Traffic Calming")
            source = st.selectbox("Data Reporting Source", sorted(meta['source_to_code'].keys()))
            st.caption(
                "ℹ️ This reflects *which data provider* reported the accident, not a real-world "
                "cause. Our EDA found severity labeling varies sharply by source (8% vs ~33% "
                "high-severity rate). Included for model transparency — see README."
            )

    weekday_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    weekday_num = weekday_map[weekday_name]
    is_weekend = int(weekday_num >= 5)
    is_rush_hour = int(hour in [7, 8, 9, 16, 17, 18, 19])
    is_late_night = int(hour in [0, 1, 2, 3, 4, 5])

    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        # Build a feature row matching EXACTLY the columns the model was trained on
        row = {col: 0 for col in meta['feature_columns']}

        row['Start_Lat'] = meta['numeric_defaults']['Start_Lat']
        row['Start_Lng'] = meta['numeric_defaults']['Start_Lng']
        row['Temperature(F)'] = temperature
        row['Humidity(%)'] = humidity
        row['Pressure(in)'] = pressure
        row['Visibility(mi)'] = visibility
        row['Wind_Speed(mph)'] = wind_speed
        row['Precipitation(in)'] = precipitation
        row['Hour'] = hour
        row['Month'] = month
        row['Weekday_Num'] = weekday_num
        row['Is_Weekend'] = is_weekend
        row['Is_Rush_Hour'] = is_rush_hour
        row['Is_Late_Night'] = is_late_night
        row['Is_Day'] = int(is_day)
        row['State_Encoded'] = meta['state_to_code'][state]
        row['Source_Encoded'] = meta['source_to_code'][source]

        row['Junction'] = junction
        row['Traffic_Signal'] = traffic_signal
        row['Crossing'] = crossing
        row['Stop'] = stop
        row['Station'] = station
        row['Amenity'] = amenity
        row['Bump'] = bump
        row['Give_Way'] = give_way
        row['No_Exit'] = no_exit
        row['Railway'] = railway
        row['Traffic_Calming'] = traffic_calming

        weather_col = f"Weather_{weather}"
        if weather_col in row:
            row[weather_col] = True

        wind_col = f"Wind_{wind_dir}"
        if wind_col in row:
            row[wind_col] = True

        X_input = pd.DataFrame([row])[meta['feature_columns']]

        prob = model.predict_proba(X_input)[0, 1]
        threshold = config['threshold']
        is_high_risk = prob >= threshold

        st.divider()
        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            st.metric("High-Severity Risk Probability", f"{prob * 100:.1f}%")
            if is_high_risk:
                st.error(f"⚠️ **HIGH RISK** (threshold: {threshold * 100:.0f}%)")
            else:
                st.success(f"✅ **LOW RISK** (threshold: {threshold * 100:.0f}%)")
            st.caption(f"Model: XGBoost · Decision threshold tuned for 90% recall on severe accidents")

        with result_col2:
            st.markdown("**Why this prediction? (SHAP explanation)**")
            shap_vals = explainer.shap_values(X_input)[0]
            feature_vals = X_input.iloc[0]

            contrib = pd.DataFrame({
                'Feature': meta['feature_columns'],
                'Value': feature_vals.values,
                'SHAP': shap_vals
            }).sort_values('SHAP', ascending=False)

            top_increase = contrib.head(4)
            top_decrease = contrib.tail(4).sort_values('SHAP')

            inc_col, dec_col = st.columns(2)
            with inc_col:
                st.markdown("🔺 *Increasing risk*")
                for _, r in top_increase.iterrows():
                    if r['SHAP'] > 0:
                        st.write(f"`{r['Feature']}` = {r['Value']:.2f}  (+{r['SHAP']:.3f})")
            with dec_col:
                st.markdown("🔻 *Decreasing risk*")
                for _, r in top_decrease.iterrows():
                    if r['SHAP'] < 0:
                        st.write(f"`{r['Feature']}` = {r['Value']:.2f}  ({r['SHAP']:.3f})")

# ============================================================
# TAB 2: ACCIDENT HEATMAP
# ============================================================
with tab2:
    st.header("Geographic Risk Distribution")

    @st.cache_data
    def load_heatmap_data():
        sample = pd.read_parquet("data/dashboard_sample.parquet")
        with open("data/state_risk_table.json") as f:
            state_table = pd.DataFrame(json.load(f))
        return sample, state_table

    sample, state_table = load_heatmap_data()

    st.warning(
        "⚠️ **Data quality note:** State-level risk differences partially reflect "
        "differences in *data provider reporting methodology*, not purely road danger. "
        "See the Source-bias finding in the README before drawing strong conclusions "
        "about any single state."
    )

    map_col, rank_col = st.columns([2, 1])

    with map_col:
        st.subheader("Accident Density (sample of 20,000)")
        color_by = st.radio("Color points by:", ["Severity", "Risk_Target"], horizontal=True)
        st.map(
            sample.rename(columns={"Start_Lat": "lat", "Start_Lng": "lon"})[["lat", "lon"]],
            size=20
        )

    with rank_col:
        st.subheader("Highest-Risk States")
        st.caption("Min. 500 accidents in sample")
        top_states = state_table.sort_values("High_Severity_Rate", ascending=False).head(10)
        st.dataframe(
            top_states[["State", "High_Severity_Rate", "Total_Accidents"]],
            hide_index=True, use_container_width=True
        )

        st.subheader("Lowest-Risk States")
        bottom_states = state_table.sort_values("High_Severity_Rate", ascending=True).head(10)
        st.dataframe(
            bottom_states[["State", "High_Severity_Rate", "Total_Accidents"]],
            hide_index=True, use_container_width=True
        )

    st.divider()
    st.subheader("Full State Comparison")
    fig_state = px.bar(
        state_table.sort_values("High_Severity_Rate", ascending=True),
        x="High_Severity_Rate", y="State", orientation="h",
        labels={"High_Severity_Rate": "High-Severity Rate (%)", "State": ""},
        height=900, color="High_Severity_Rate", color_continuous_scale="Reds"
    )
    fig_state.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_state, use_container_width=True)

# ============================================================
# TAB 3: WEATHER INFLUENCE
# ============================================================
with tab3:
    st.header("Weather & Road Feature Influence on Severity")

    @st.cache_data
    def load_weather_road_data():
        with open("data/weather_risk_table.json") as f:
            weather_table = pd.DataFrame(json.load(f))
        with open("data/road_risk_table.json") as f:
            road_table = pd.DataFrame(json.load(f))
        return weather_table, road_table

    weather_table, road_table = load_weather_road_data()

    st.info(
        "💡 **Counter-intuitive finding from EDA:** clear, calm weather shows *higher* "
        "severity rates than fog, snow, or storms. The likely explanation — drivers slow "
        "down when conditions visibly demand caution, but not in ordinary clear weather. "
        "When crashes do happen in 'easy' conditions, they happen at higher speed."
    )

    weather_col, road_col = st.columns(2)

    with weather_col:
        st.subheader("🌦️ High-Severity Rate by Weather Condition")
        st.caption("Min. 500 occurrences in sample")
        weather_sorted = weather_table.sort_values("High_Severity_Rate", ascending=True)
        fig_weather = px.bar(
            weather_sorted, x="High_Severity_Rate", y="Weather_Condition", orientation="h",
            labels={"High_Severity_Rate": "High-Severity Rate (%)", "Weather_Condition": ""},
            height=550, color="High_Severity_Rate", color_continuous_scale="Oranges",
            hover_data={"Total_Accidents": True}
        )
        fig_weather.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_weather, use_container_width=True)

    with road_col:
        st.subheader("🛣️ Severity: Feature Present vs Absent")
        st.caption("Controlled features (signals, stops) reduce severity; uncontrolled junctions increase it")
        road_melted = road_table.melt(
            id_vars="Feature", value_vars=["Severity_When_Present", "Severity_When_Absent"],
            var_name="Condition", value_name="High_Severity_Rate"
        )
        road_melted["Condition"] = road_melted["Condition"].map({
            "Severity_When_Present": "Present", "Severity_When_Absent": "Absent"
        })
        fig_road = px.bar(
            road_melted, x="High_Severity_Rate", y="Feature", color="Condition",
            orientation="h", barmode="group", height=550,
            labels={"High_Severity_Rate": "High-Severity Rate (%)", "Feature": ""},
            color_discrete_map={"Present": "#EF553B", "Absent": "#636EFA"}
        )
        st.plotly_chart(fig_road, use_container_width=True)

    st.divider()
    st.markdown(
        "**Reading this tab:** `Junction` is the one road feature where severity is *higher* "
        "when present — likely because junctions include uncontrolled intersections and merge "
        "points where vehicles cross paths at speed, unlike signal-/stop-controlled crossings "
        "which force a slowdown."
    )

# ============================================================
# TAB 4: FEATURE IMPORTANCE
# ============================================================
with tab4:
    st.header("Global Feature Importance (SHAP)")

    @st.cache_data
    def load_importance_data():
        with open("data/feature_importance.json") as f:
            return pd.DataFrame(json.load(f))

    importance_table = load_importance_data()

    st.warning(
        "⚠️ **`Source_Encoded` is the top feature** — this reflects which data provider "
        "reported the accident, not a real-world danger factor. Our EDA found this source "
        "has a 4x swing in reported severity rate by itself (8% vs ~33%), independent of "
        "actual road conditions. See README for the full investigation."
    )

    fig_importance = px.bar(
        importance_table.sort_values("Mean_Abs_SHAP", ascending=True),
        x="Mean_Abs_SHAP", y="Feature", orientation="h",
        labels={"Mean_Abs_SHAP": "Mean |SHAP value| (avg. impact on prediction)", "Feature": ""},
        height=700, color="Mean_Abs_SHAP", color_continuous_scale="Purples"
    )
    fig_importance.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_importance, use_container_width=True)

    st.divider()
    st.markdown(
        """
        **How to read this chart:** each bar shows how much, *on average*, that feature moved
        the model's prediction up or down across 2,000 sample test accidents — regardless of
        direction. A long bar means the feature matters a lot for *some* predictions; it doesn't
        tell you whether it pushes risk up or down (use the Risk Calculator's per-prediction SHAP
        panel for that, since the direction can depend on the specific scenario).

        **Notable patterns confirmed here, matching our EDA:**
        - `Traffic_Signal`, `Crossing`, `Stop` — controlled road features, consistently reduce severity
        - `Junction` — the one road feature that *increases* severity (uncontrolled intersections/merges)
        - `Start_Lat` / `Start_Lng` — strong signal, but a known overfitting risk (see README limitations)
        """
    )
