import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import plotly.express as px

st.set_page_config(page_title="Traffic Accident Risk Predictor", layout="wide", page_icon=":vertical_traffic_light:")


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

st.title("Traffic Accident Severity Risk Predictor")
st.caption(
    "Predicts whether a US traffic accident is likely to be high-severity "
    "(injury/fatality-level) based on conditions at the time and place. "
    "Trained on 500K+ records from the US-Accidents dataset (Moosavi et al.)."
)

with st.expander("Methodology notes"):
    st.markdown(
        "This model predicts severity given that an accident occurred, not whether one "
        "will happen; the dataset only contains accidents that already happened. "
        "Severity reporting also varies by data source (roughly 8% vs 33% high-severity "
        "rate between sources), so `Source` is included as a model feature rather than "
        "removed. Full writeup is in the project README."
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["Risk Calculator", "Accident Heatmap", "Weather Influence", "Feature Importance"]
)


with tab1:
    st.header("Check a Scenario")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Location & Time")
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
        st.subheader("Weather")
        weather = st.selectbox("Weather Condition", meta['weather_categories'])
        wind_dir = st.selectbox("Wind Direction", meta['wind_categories'])
        temperature = st.number_input("Temperature (°F)", value=meta['numeric_defaults']['Temperature(F)'])
        humidity = st.number_input("Humidity (%)", value=meta['numeric_defaults']['Humidity(%)'], min_value=0.0, max_value=100.0)

    with col3:
        st.subheader("Road Features")
        junction = st.checkbox("Junction")
        traffic_signal = st.checkbox("Traffic Signal")
        crossing = st.checkbox("Crossing")
        stop = st.checkbox("Stop Sign")
        station = st.checkbox("Station nearby")

    with st.expander("Advanced: weather detail, more road features, data source"):
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
            st.caption("Which provider reported the accident. See Methodology notes above.")

    weekday_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    weekday_num = weekday_map[weekday_name]
    is_weekend = int(weekday_num >= 5)
    is_rush_hour = int(hour in [7, 8, 9, 16, 17, 18, 19])
    is_late_night = int(hour in [0, 1, 2, 3, 4, 5])

    if st.button("Predict Risk", type="primary", use_container_width=True):
      
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
                st.error(f"HIGH RISK (threshold: {threshold * 100:.0f}%)")
            else:
                st.success(f"LOW RISK (threshold: {threshold * 100:.0f}%)")
            st.caption("Model: XGBoost. Threshold tuned for 90% recall on severe accidents.")

        with result_col2:
            st.markdown("**Why this prediction**")
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
                st.markdown("Increasing risk")
                for _, r in top_increase.iterrows():
                    if r['SHAP'] > 0:
                        st.write(f"`{r['Feature']}` = {r['Value']:.2f}  (+{r['SHAP']:.3f})")
            with dec_col:
                st.markdown("Decreasing risk")
                for _, r in top_decrease.iterrows():
                    if r['SHAP'] < 0:
                        st.write(f"`{r['Feature']}` = {r['Value']:.2f}  ({r['SHAP']:.3f})")


with tab2:
    st.header("Geographic Risk Distribution")
    st.caption("State differences include a reporting-source effect. See Methodology notes above.")

    @st.cache_data
    def load_heatmap_data():
        sample = pd.read_parquet("data/dashboard_sample.parquet")
        with open("data/state_risk_table.json") as f:
            state_table = pd.DataFrame(json.load(f))
        return sample, state_table

    sample, state_table = load_heatmap_data()

    map_col, rank_col = st.columns([2, 1])

    with map_col:
        st.subheader("Accident Density (sample of 20,000)")
        color_by = st.radio("Color points by:", ["Severity", "Risk_Target"], horizontal=True)

        severity_colors = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c"}
        risk_colors = {0: "#3498db", 1: "#e74c3c"}

        map_data = sample.rename(columns={"Start_Lat": "lat", "Start_Lng": "lon"}).copy()
        if color_by == "Severity":
            map_data["color"] = map_data["Severity"].map(severity_colors)
            st.caption("Green (Severity 1) to red (Severity 4)")
        else:
            map_data["color"] = map_data["Risk_Target"].map(risk_colors)
            st.caption("Blue = Low-Risk, Red = High-Risk")

        st.map(map_data[["lat", "lon", "color"]], size=20, color="color")

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


with tab3:
    st.header("Weather & Road Feature Influence on Severity")
    st.caption(
        "Clear, calm weather shows higher severity than fog, snow, or storms in this data "
        "- drivers slow down for visibly bad conditions but not for ordinary clear weather."
    )

    @st.cache_data
    def load_weather_road_data():
        with open("data/weather_risk_table.json") as f:
            weather_table = pd.DataFrame(json.load(f))
        with open("data/road_risk_table.json") as f:
            road_table = pd.DataFrame(json.load(f))
        return weather_table, road_table

    weather_table, road_table = load_weather_road_data()

    weather_col, road_col = st.columns(2)

    with weather_col:
        st.subheader("High-Severity Rate by Weather Condition")
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
        st.subheader("Severity: Feature Present vs Absent")
        st.caption("Signals/stops reduce severity; uncontrolled junctions increase it")
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

    st.caption(
        "Junction is the one road feature where severity is higher when present, likely "
        "uncontrolled intersections and merge points rather than signal- or stop-controlled ones."
    )


with tab4:
    st.header("Global Feature Importance (SHAP)")
    st.caption(
        "Source_Encoded ranks highest - it reflects which provider reported the accident, "
        "not a physical danger factor. See Methodology notes above."
    )

    @st.cache_data
    def load_importance_data():
        with open("data/feature_importance.json") as f:
            return pd.DataFrame(json.load(f))

    importance_table = load_importance_data()

    fig_importance = px.bar(
        importance_table.sort_values("Mean_Abs_SHAP", ascending=True),
        x="Mean_Abs_SHAP", y="Feature", orientation="h",
        labels={"Mean_Abs_SHAP": "Mean |SHAP value|", "Feature": ""},
        height=700, color="Mean_Abs_SHAP", color_continuous_scale="Purples"
    )
    fig_importance.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_importance, use_container_width=True)

    st.caption(
        "Bar length is average impact magnitude, not direction - it doesn't say whether a "
        "feature pushes risk up or down. Use the Risk Calculator's per-prediction panel for that."
    )
