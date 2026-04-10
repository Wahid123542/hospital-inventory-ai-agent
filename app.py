import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Hospital Inventory Intelligence Agent",
    layout="wide",
)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("hospital_inventory_data.csv")

# -----------------------------
# Add medication categories
# -----------------------------
medication_category_map = {
    "Amoxicillin": "Antibiotic",
    "Ibuprofen": "Pain Relief",
    "Paracetamol": "Pain Relief",
    "Insulin": "Diabetes",
    "Metformin": "Diabetes",
    "Atorvastatin": "Cardiovascular",
    "Ceftriaxone": "Antibiotic",
    "Azithromycin": "Antibiotic",
    "Heparin": "Anticoagulant",
    "Morphine": "Critical Care",
}

df["medication_category"] = df["medication_name"].map(medication_category_map)

# -----------------------------
# Derive season from month
# -----------------------------
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

df["season"] = df["month"].apply(get_season)

# -----------------------------
# Learn seasonality from data
# -----------------------------
overall_baseline_usage = df["daily_usage"].mean()

season_avg_usage = df.groupby("season")["daily_usage"].mean()
season_multipliers = (season_avg_usage / overall_baseline_usage).to_dict()

category_baseline_usage = (
    df.groupby("medication_category")["daily_usage"].mean().to_dict()
)

season_category_avg_usage = (
    df.groupby(["season", "medication_category"])["daily_usage"]
    .mean()
    .reset_index()
)

season_category_multiplier_map = {}
for _, row in season_category_avg_usage.iterrows():
    season_name = row["season"]
    category = row["medication_category"]
    baseline_cat = category_baseline_usage[category]
    multiplier = row["daily_usage"] / baseline_cat if baseline_cat != 0 else 1.0
    season_category_multiplier_map[(season_name, category)] = multiplier

# fill any missing season-category combos with 1.0
all_seasons = ["Winter", "Spring", "Summer", "Fall"]
all_categories = sorted(df["medication_category"].dropna().unique().tolist())
for s in all_seasons:
    for c in all_categories:
        if (s, c) not in season_category_multiplier_map:
            season_category_multiplier_map[(s, c)] = 1.0

# -----------------------------
# Sidebar filters + simulation
# -----------------------------
st.sidebar.title("Inventory Filters")

departments = ["All"] + sorted(df["department"].unique().tolist())
selected_department = st.sidebar.selectbox("Department", departments)

medications = ["All"] + sorted(df["medication_name"].unique().tolist())
selected_medication = st.sidebar.selectbox("Medication", medications)

categories = ["All"] + sorted(df["medication_category"].unique().tolist())
selected_category = st.sidebar.selectbox("Medication Category", categories)

st.sidebar.markdown("---")
st.sidebar.subheader("Seasonal Scenario Simulation")

selected_season = st.sidebar.selectbox("Season Scenario", ["Winter", "Spring", "Summer", "Fall"])
seasonal_severity = st.sidebar.slider("Seasonal Illness Severity", 0.5, 2.0, 1.0, 0.1)
patient_multiplier = st.sidebar.slider("Patient Volume Multiplier", 0.5, 2.0, 1.0, 0.1)
delay_multiplier = st.sidebar.slider("Supplier Delay Multiplier", 1.0, 2.0, 1.0, 0.1)

# -----------------------------
# Apply data-driven season effect
# -----------------------------
df["selected_season"] = selected_season
df["learned_season_multiplier"] = df["medication_category"].apply(
    lambda c: season_category_multiplier_map.get((selected_season, c), 1.0)
)

# scenario severity acts as an override on top of learned seasonality
df["scenario_season_effect"] = df["learned_season_multiplier"] * seasonal_severity

# -----------------------------
# Core inventory logic
# -----------------------------
df["adjusted_supplier_lead_time_days"] = (
    df["supplier_lead_time_days"] * delay_multiplier
).round(1)

df["days_remaining"] = df["current_stock"] / df["daily_usage"]

def stock_status(row):
    if row["current_stock"] < row["reorder_threshold"]:
        return "LOW"
    elif row["days_remaining"] < row["adjusted_supplier_lead_time_days"]:
        return "RISK"
    else:
        return "OK"

def reorder_recommendation(row):
    if row["stock_status"] == "LOW":
        return "Reorder Immediately"
    elif row["stock_status"] == "RISK":
        return "Reorder Soon"
    else:
        return "Stock OK"

df["stock_status"] = df.apply(stock_status, axis=1)
df["action"] = df.apply(reorder_recommendation, axis=1)

# -----------------------------
# Forecasting logic
# -----------------------------
np.random.seed(42)

random_noise = np.random.normal(1.0, 0.08, size=len(df))

df["future_month_demand"] = (
    df["daily_usage"]
    * 30
    * df["scenario_season_effect"]
    * patient_multiplier
    * random_noise
).round(0)

df_model = df.copy()
df_encoded = pd.get_dummies(
    df_model,
    columns=["department", "medication_name", "medication_category", "season"],
    drop_first=True
)

X = df_encoded.drop(
    columns=[
        "expiration_date",
        "future_month_demand",
        "stock_status",
        "action",
        "selected_season",
    ]
)
y = df_encoded["future_month_demand"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

df["predicted_month_demand"] = model.predict(X).round(0)
df["projected_days_remaining"] = df["current_stock"] / (df["predicted_month_demand"] / 30)

def future_risk(row):
    if row["projected_days_remaining"] < row["adjusted_supplier_lead_time_days"]:
        return "Future Shortage Risk"
    return "Stable"

df["future_risk"] = df.apply(future_risk, axis=1)

# -----------------------------
# Agent logic
# -----------------------------
def priority_level(row):
    if row["stock_status"] == "LOW" and row["future_risk"] == "Future Shortage Risk":
        return "HIGH"
    elif row["stock_status"] == "LOW" or row["future_risk"] == "Future Shortage Risk":
        return "MEDIUM"
    elif row["stock_status"] == "RISK":
        return "MEDIUM"
    return "LOW"

def agent_reason(row):
    reasons = []

    if row["stock_status"] == "LOW":
        reasons.append("current stock is below reorder threshold")

    if row["stock_status"] == "RISK":
        reasons.append("current supply may not cover adjusted supplier lead time")

    if row["future_risk"] == "Future Shortage Risk":
        reasons.append("predicted future demand may cause a shortage")

    if row["days_remaining"] < 7:
        reasons.append("less than 7 days of stock remain")

    learned_pct = (row["learned_season_multiplier"] - 1.0) * 100
    if abs(learned_pct) >= 1:
        reasons.append(
            f"historical data suggests {selected_season.lower()} changes {row['medication_category'].lower()} demand by about {learned_pct:.0f}%"
        )

    if seasonal_severity != 1.0:
        reasons.append("seasonal severity scenario further adjusts expected demand")

    if patient_multiplier > 1.0:
        reasons.append("higher patient volume scenario increases demand")

    if delay_multiplier > 1.0:
        reasons.append("supplier delay scenario increases replenishment risk")

    if not reasons:
        return "inventory position is stable"

    return "; ".join(reasons)

def recommended_order_qty(row):
    target_days = 30
    target_stock = (row["predicted_month_demand"] / 30) * target_days
    qty = max(0, round(target_stock - row["current_stock"]))
    return qty

def agent_action(row):
    if row["priority"] == "HIGH":
        return "Reorder immediately and notify supply manager"
    elif row["priority"] == "MEDIUM":
        return "Review inventory and prepare reorder soon"
    return "Continue monitoring"

df["priority"] = df.apply(priority_level, axis=1)
df["agent_reason"] = df.apply(agent_reason, axis=1)
df["recommended_order_qty"] = df.apply(recommended_order_qty, axis=1)
df["agent_action"] = df.apply(agent_action, axis=1)

# -----------------------------
# Formatting
# -----------------------------
df["days_remaining"] = df["days_remaining"].round(1)
df["projected_days_remaining"] = df["projected_days_remaining"].round(1)
df["predicted_month_demand"] = df["predicted_month_demand"].round(0)
df["learned_season_multiplier"] = df["learned_season_multiplier"].round(2)
df["scenario_season_effect"] = df["scenario_season_effect"].round(2)

# -----------------------------
# Additional filters
# -----------------------------
status_options = ["All"] + sorted(df["stock_status"].unique().tolist())
selected_status = st.sidebar.selectbox("Current Stock Status", status_options)

priority_options = ["All"] + sorted(df["priority"].unique().tolist())
selected_priority = st.sidebar.selectbox("Priority Level", priority_options)

filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[filtered_df["department"] == selected_department]

if selected_medication != "All":
    filtered_df = filtered_df[filtered_df["medication_name"] == selected_medication]

if selected_category != "All":
    filtered_df = filtered_df[filtered_df["medication_category"] == selected_category]

if selected_status != "All":
    filtered_df = filtered_df[filtered_df["stock_status"] == selected_status]

if selected_priority != "All":
    filtered_df = filtered_df[filtered_df["priority"] == selected_priority]

# -----------------------------
# Header
# -----------------------------
st.title("🏥 Hospital Inventory Intelligence Agent")
st.markdown(
    """
    This dashboard monitors medication stock, predicts future demand,
    uses learned seasonality from historical data, and recommends actions before shortages happen.
    """
)

# -----------------------------
# Active simulation scenario
# -----------------------------
st.subheader("🧪 Active Seasonal Scenario")

sim1, sim2, sim3, sim4 = st.columns(4)
sim1.info(f"Season Scenario: **{selected_season}**")
sim2.warning(f"Seasonal Severity: **{seasonal_severity:.1f}x**")
sim3.success(f"Patient Volume: **{patient_multiplier:.1f}x**")
sim4.error(f"Supplier Delay: **{delay_multiplier:.1f}x**")

st.markdown("---")

# -----------------------------
# Learned seasonality summary
# -----------------------------
st.subheader("📚 Learned Seasonality from Data")

season_summary_df = pd.DataFrame({
    "Season": list(season_multipliers.keys()),
    "Learned Multiplier": [round(v, 2) for v in season_multipliers.values()]
}).sort_values("Season")

col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    st.dataframe(season_summary_df, use_container_width=True)

with col_s2:
    fig_learned = px.bar(
        season_summary_df,
        x="Season",
        y="Learned Multiplier",
        color="Season",
        text="Learned Multiplier",
        title="Historical Seasonal Demand Multipliers"
    )
    fig_learned.update_traces(textposition="outside")
    fig_learned.update_layout(height=360, yaxis_title="Multiplier")
    st.plotly_chart(fig_learned, use_container_width=True)

st.caption(
    "A multiplier of 1.00 means baseline demand. Above 1.00 means that season historically shows higher demand than average."
)

st.markdown("---")

# -----------------------------
# Top metrics
# -----------------------------
total_records = len(filtered_df)
low_count = (filtered_df["stock_status"] == "LOW").sum()
risk_count = (filtered_df["stock_status"] == "RISK").sum()
future_risk_count = (filtered_df["future_risk"] == "Future Shortage Risk").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", total_records)
col2.metric("Low Stock", low_count)
col3.metric("At Risk", risk_count)
col4.metric("Future Shortage Risk", future_risk_count)

st.markdown("---")

# -----------------------------
# Quick insights
# -----------------------------
st.subheader("🔍 Quick Insights")

if not filtered_df.empty:
    top_med_usage = (
        filtered_df.groupby("medication_name")["daily_usage"]
        .mean()
        .sort_values(ascending=False)
    )
    top_med_demand = (
        filtered_df.groupby("medication_name")["predicted_month_demand"]
        .mean()
        .sort_values(ascending=False)
    )
    top_dept = (
        filtered_df.groupby("department")["daily_usage"]
        .sum()
        .sort_values(ascending=False)
    )
    top_category = (
        filtered_df.groupby("medication_category")["predicted_month_demand"]
        .mean()
        .sort_values(ascending=False)
    )

    insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
    insight_col1.info(f"Highest average usage medication: **{top_med_usage.index[0]}**")
    insight_col2.warning(f"Highest predicted demand medication: **{top_med_demand.index[0]}**")
    insight_col3.success(f"Highest usage department: **{top_dept.index[0]}**")
    insight_col4.error(f"Highest demand category: **{top_category.index[0]}**")

st.markdown("---")

# -----------------------------
# Agent insights
# -----------------------------
st.subheader("🧠 Agent Insights")

agent_view = filtered_df.copy()

high_priority = agent_view[agent_view["priority"] == "HIGH"]
medium_priority = agent_view[agent_view["priority"] == "MEDIUM"]

col_a, col_b, col_c = st.columns(3)
col_a.metric("High Priority Items", len(high_priority))
col_b.metric("Medium Priority Items", len(medium_priority))
col_c.metric("Recommended Reorder Units", int(agent_view["recommended_order_qty"].sum()))

if len(high_priority) > 0:
    st.error(f"Immediate attention needed: {len(high_priority)} item(s) are high priority.")
elif len(medium_priority) > 0:
    st.warning(f"{len(medium_priority)} item(s) need review soon.")
else:
    st.success("No urgent inventory actions needed right now.")

st.subheader("Top Agent Recommendations")

priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
top_actions = agent_view.copy()
top_actions["priority_rank"] = top_actions["priority"].map(priority_order)
top_actions = top_actions.sort_values(by=["priority_rank", "projected_days_remaining"])

st.dataframe(
    top_actions[
        [
            "medication_name",
            "medication_category",
            "department",
            "priority",
            "current_stock",
            "daily_usage",
            "days_remaining",
            "predicted_month_demand",
            "projected_days_remaining",
            "learned_season_multiplier",
            "recommended_order_qty",
            "agent_reason",
            "agent_action",
        ]
    ].head(15),
    use_container_width=True,
)

st.subheader("📋 Agent Summary")

if not agent_view.empty:
    top_med = (
        agent_view.groupby("medication_name")["predicted_month_demand"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    top_dept = (
        agent_view.groupby("department")["daily_usage"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    top_cat = (
        agent_view.groupby("medication_category")["predicted_month_demand"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    learned_season_value = season_multipliers.get(selected_season, 1.0)

    st.write(
        f"""
        The agent reviewed the filtered inventory and found **{len(high_priority)} high-priority items**
        and **{len(medium_priority)} medium-priority items**.

        Historical data suggests **{selected_season}** runs at about **{learned_season_value:.2f}x**
        baseline demand overall.

        Under the current scenario, the highest predicted-demand medication is **{top_med}**.

        The medication category with the highest projected demand is **{top_cat}**.

        The department with the highest overall usage is **{top_dept}**.
        """
    )

st.markdown("---")

# -----------------------------
# Ask the Agent
# -----------------------------
st.subheader("💬 Ask the Agent")

user_question = st.text_input("Ask a question about inventory, risk, or seasonality")

if user_question:
    question = user_question.lower()

    if "season" in question or "winter" in question or "summer" in question or "fall" in question or "spring" in question:
        st.write("### Agent Answer")
        learned_season_value = season_multipliers.get(selected_season, 1.0)
        st.write(
            f"""
            The active season scenario is **{selected_season}**.

            Based on historical data, **{selected_season}** has an overall learned multiplier of
            **{learned_season_value:.2f}x** relative to baseline demand.

            The current severity setting is **{seasonal_severity:.1f}x**, which further adjusts
            that learned seasonal effect.
            """
        )

    elif "category" in question and "highest" in question:
        top_cat_df = (
            filtered_df.groupby("medication_category")["predicted_month_demand"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        st.write("### Agent Answer")
        if top_cat_df.empty:
            st.warning("No category data available.")
        else:
            top_cat = top_cat_df.iloc[0]
            st.write(
                f"**{top_cat['medication_category']}** has the highest predicted demand "
                f"at about **{top_cat['predicted_month_demand']:.0f}** units."
            )
            st.dataframe(top_cat_df, use_container_width=True)

    elif "reorder" in question or "order" in question:
        reorder_items = filtered_df[
            filtered_df["priority"].isin(["HIGH", "MEDIUM"])
        ][
            [
                "medication_name",
                "medication_category",
                "department",
                "priority",
                "current_stock",
                "recommended_order_qty",
                "agent_action",
            ]
        ].head(10)

        st.write("### Agent Answer")
        if reorder_items.empty:
            st.success("No medications currently need reorder action.")
        else:
            st.write("These medications should be reviewed for reorder first:")
            st.dataframe(reorder_items, use_container_width=True)

    elif "highest demand" in question or "top demand" in question:
        top_demand = (
            filtered_df.groupby("medication_name")["predicted_month_demand"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        st.write("### Agent Answer")
        if top_demand.empty:
            st.warning("No data available for predicted demand.")
        else:
            top_med = top_demand.iloc[0]
            st.write(
                f"**{top_med['medication_name']}** has the highest predicted monthly demand "
                f"at about **{top_med['predicted_month_demand']:.0f}** units under the current scenario."
            )
            st.dataframe(top_demand.head(10), use_container_width=True)

    elif "highest usage" in question or "most used" in question:
        top_usage = (
            filtered_df.groupby("medication_name")["daily_usage"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        st.write("### Agent Answer")
        if top_usage.empty:
            st.warning("No usage data available.")
        else:
            top_med = top_usage.iloc[0]
            st.write(
                f"**{top_med['medication_name']}** has the highest average daily usage "
                f"at about **{top_med['daily_usage']:.1f}** units per day."
            )
            st.dataframe(top_usage.head(10), use_container_width=True)

    elif "department" in question and ("most" in question or "highest" in question):
        dept_usage = (
            filtered_df.groupby("department")["daily_usage"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        st.write("### Agent Answer")
        if dept_usage.empty:
            st.warning("No department data available.")
        else:
            top_dept = dept_usage.iloc[0]
            st.write(
                f"**{top_dept['department']}** has the highest medication usage "
                f"with **{top_dept['daily_usage']:.0f}** total units."
            )
            st.dataframe(dept_usage, use_container_width=True)

    elif "low stock" in question or "low" in question:
        low_items = filtered_df[
            filtered_df["stock_status"] == "LOW"
        ][
            [
                "medication_name",
                "medication_category",
                "department",
                "current_stock",
                "reorder_threshold",
                "priority",
                "agent_action",
            ]
        ].head(10)

        st.write("### Agent Answer")
        if low_items.empty:
            st.success("There are no low-stock items in the current filtered view.")
        else:
            st.write("These items are currently below reorder threshold:")
            st.dataframe(low_items, use_container_width=True)

    elif "high priority" in question or "priority" in question:
        priority_items = filtered_df[
            filtered_df["priority"] == "HIGH"
        ][
            [
                "medication_name",
                "medication_category",
                "department",
                "current_stock",
                "predicted_month_demand",
                "recommended_order_qty",
                "agent_reason",
                "agent_action",
            ]
        ].head(10)

        st.write("### Agent Answer")
        if priority_items.empty:
            st.success("No high-priority items found in the current filtered view.")
        else:
            st.write("These are the highest-priority items right now:")
            st.dataframe(priority_items, use_container_width=True)

    elif "summary" in question or "overall" in question:
        low_count = (filtered_df["stock_status"] == "LOW").sum()
        risk_count = (filtered_df["stock_status"] == "RISK").sum()
        future_risk_count = (filtered_df["future_risk"] == "Future Shortage Risk").sum()

        st.write("### Agent Answer")
        st.write(
            f"""
            Here is the current summary:
            - **{low_count}** items are low stock
            - **{risk_count}** items are at immediate risk
            - **{future_risk_count}** items have future shortage risk
            - Active season: **{selected_season}**
            - Learned season multiplier: **{season_multipliers.get(selected_season, 1.0):.2f}x**
            - Seasonal severity: **{seasonal_severity:.1f}x**
            - Patient volume multiplier: **{patient_multiplier:.1f}x**
            - Supplier delay multiplier: **{delay_multiplier:.1f}x**
            """
        )

    else:
        st.write("### Agent Answer")
        st.info(
            "I can answer questions like: "
            "'What should I reorder now?', "
            "'Which medication has the highest predicted demand?', "
            "'Which category has the highest demand?', "
            "'Which department uses the most medication?', "
            "'Show me high priority items.', "
            "'Give me a summary.', "
            "or 'What does the season multiplier mean?'"
        )

st.markdown("---")

# -----------------------------
# Critical alerts
# -----------------------------
st.subheader("🚨 Critical Alerts")

critical_items = filtered_df[
    (filtered_df["stock_status"].isin(["LOW", "RISK"])) |
    (filtered_df["future_risk"] == "Future Shortage Risk")
].copy()

critical_items = critical_items.sort_values(
    by=["projected_days_remaining", "current_stock"],
    ascending=[True, True]
)

if critical_items.empty:
    st.success("No critical alerts for the selected filters.")
else:
    st.dataframe(
        critical_items[
            [
                "medication_name",
                "medication_category",
                "department",
                "current_stock",
                "daily_usage",
                "days_remaining",
                "predicted_month_demand",
                "projected_days_remaining",
                "stock_status",
                "future_risk",
                "action",
            ]
        ],
        use_container_width=True,
    )

st.markdown("---")

# -----------------------------
# Charts row 1
# -----------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📊 Current Stock Status")
    status_counts = filtered_df["stock_status"].value_counts().reset_index()
    status_counts.columns = ["stock_status", "count"]

    fig_status = px.bar(
        status_counts,
        x="stock_status",
        y="count",
        color="stock_status",
        text="count",
        color_discrete_map={
            "LOW": "#EF4444",
            "RISK": "#F59E0B",
            "OK": "#10B981",
        },
        title="Current Inventory Health"
    )
    fig_status.update_traces(textposition="outside")
    fig_status.update_layout(height=420)
    st.plotly_chart(fig_status, use_container_width=True)

with chart_col2:
    st.subheader("💊 Medication Usage")
    med_usage = (
        filtered_df.groupby("medication_name")["daily_usage"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig_med_usage = px.bar(
        med_usage,
        x="medication_name",
        y="daily_usage",
        color="daily_usage",
        color_continuous_scale="Blues",
        text="daily_usage",
        title="Average Daily Usage by Medication"
    )
    fig_med_usage.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_med_usage.update_layout(height=420, xaxis_title="Medication", yaxis_title="Avg Daily Usage")
    st.plotly_chart(fig_med_usage, use_container_width=True)

# -----------------------------
# Charts row 2
# -----------------------------
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("📈 Predicted Monthly Demand")
    pred_demand = (
        filtered_df.groupby("medication_name")["predicted_month_demand"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig_pred = px.bar(
        pred_demand,
        x="medication_name",
        y="predicted_month_demand",
        color="predicted_month_demand",
        color_continuous_scale="Purples",
        text="predicted_month_demand",
        title="Predicted Demand Next Month"
    )
    fig_pred.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_pred.update_layout(height=420, xaxis_title="Medication", yaxis_title="Predicted Monthly Demand")
    st.plotly_chart(fig_pred, use_container_width=True)

with chart_col4:
    st.subheader("🏷️ Demand by Medication Category")
    cat_demand = (
        filtered_df.groupby("medication_category")["predicted_month_demand"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig_cat = px.bar(
        cat_demand,
        x="medication_category",
        y="predicted_month_demand",
        color="medication_category",
        text="predicted_month_demand",
        title="Average Predicted Demand by Category"
    )
    fig_cat.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_cat.update_layout(height=420, xaxis_title="Category", yaxis_title="Predicted Monthly Demand")
    st.plotly_chart(fig_cat, use_container_width=True)

# -----------------------------
# Charts row 3
# -----------------------------
chart_col5, chart_col6 = st.columns(2)

with chart_col5:
    st.subheader("🏥 Department Usage")
    dept_usage = (
        filtered_df.groupby("department")["daily_usage"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig_dept = px.pie(
        dept_usage,
        names="department",
        values="daily_usage",
        title="Share of Medication Usage by Department",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_dept.update_layout(height=420)
    st.plotly_chart(fig_dept, use_container_width=True)

with chart_col6:
    st.subheader("⏳ Current Stock vs Reorder Threshold")

    comparison_df = (
        filtered_df.groupby("medication_name")[["current_stock", "reorder_threshold"]]
        .mean()
        .reset_index()
    )

    fig_compare = px.bar(
        comparison_df,
        x="medication_name",
        y=["current_stock", "reorder_threshold"],
        barmode="group",
        title="Average Stock Compared with Reorder Threshold",
        color_discrete_sequence=["#3B82F6", "#F97316"]
    )
    fig_compare.update_layout(height=420, yaxis_title="Units")
    st.plotly_chart(fig_compare, use_container_width=True)

st.markdown("---")

# -----------------------------
# Full table
# -----------------------------
st.subheader("🧾 Inventory Data Table")

display_df = filtered_df[
    [
        "medication_name",
        "medication_category",
        "department",
        "season",
        "current_stock",
        "daily_usage",
        "reorder_threshold",
        "days_remaining",
        "stock_status",
        "predicted_month_demand",
        "projected_days_remaining",
        "learned_season_multiplier",
        "scenario_season_effect",
        "adjusted_supplier_lead_time_days",
        "future_risk",
        "priority",
        "recommended_order_qty",
        "action",
    ]
].copy()

st.dataframe(display_df, use_container_width=True)

# -----------------------------
# Download
# -----------------------------
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_inventory_data.csv",
    mime="text/csv",
)