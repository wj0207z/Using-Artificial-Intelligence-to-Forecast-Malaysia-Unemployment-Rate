import streamlit as st
import pandas as pd
import plotly.express as px

# === Load dataset ===
df = pd.read_csv("MalaysiaQuarterlyLabourForce.csv")

# Convert 'date' to datetime and create clean string version for display
df['date'] = pd.PeriodIndex(df['date'], freq='Q').to_timestamp()
df['date'] = pd.to_datetime(df['date'])  # Keep for charting
df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')  # For clean display in table

# Extract year for filtering
df['year'] = df['date'].dt.year

# Sort by date
df = df.sort_values('date')

# === App title ===
st.title("📊 Malaysia Quarterly Labour Force Statistics (2010 - 2024)")

# === 1. Show full table (clean date for display) ===
st.subheader("Full Data Table")

# Prepare table: replace datetime with formatted date
display_table = df.copy().reset_index(drop=True)
display_table.index += 1  # Index starts at 1
cols = ['date_str'] + [col for col in display_table.columns if col not in ['date', 'date_str']]
display_table = display_table.loc[:, cols]
display_table = display_table.rename(columns={'date_str': 'date'})
st.dataframe(display_table, use_container_width=True)

# === 2. Chart Controls ===
st.subheader("Chart Controls")

metrics = {
    "Labour Force": "lf",
    "Employed": "lf_employed",
    "Unemployed": "lf_unemployed",
    "Outside Labour Force": "lf_outside",
    "Participation Rate (%)": "p_rate",
    "Employment to Population Ratio (%)": "ap_ratio",
    "Unemployment Rate (%)": "u_rate"
}
selected_metric_label = st.selectbox("Choose a metric to display:", list(metrics.keys()))
selected_metric = metrics[selected_metric_label]

# Optional: multi-metric comparison for Line Chart
multi_metric_labels = st.multiselect("Or compare multiple metrics (Line Chart only):", list(metrics.keys()))

# Chart type
chart_type = st.selectbox("Choose chart type:", ["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot"])

# Trendline for scatter plot
show_trendline = st.checkbox("📈 Show Trendline (OLS) — Scatter Plot only")

# Filter by year
year_options = sorted(df['year'].unique())
selected_years = st.multiselect("Filter by specific year(s):", options=year_options, default=year_options)

# === 3. Filter data ===
filtered_df = df[df['year'].isin(selected_years)]

# === 4. Show filtered table ===
st.subheader(f"Filtered Data Table ({len(filtered_df)} quarters)")
filtered_display = filtered_df.copy().reset_index(drop=True)
filtered_display.index += 1
cols = ['date_str'] + [col for col in filtered_display.columns if col not in ['date', 'date_str']]
filtered_display = filtered_display.loc[:, cols]
filtered_display = filtered_display.rename(columns={'date_str': 'date'})
st.dataframe(filtered_display, use_container_width=True)

# === 5. Chart Display ===
st.markdown("---")
st.subheader("📈 Labour Market Trends")

title = f"{selected_metric_label} ({', '.join(map(str, selected_years))})"

# Multi-metric line chart
if chart_type == "Line Chart" and multi_metric_labels:
    fig = px.line(
        filtered_df,
        x="date",
        y=[metrics[label] for label in multi_metric_labels],
        title="Multi-Metric Comparison Over Time"
    )
else:
    if chart_type == "Line Chart":
        fig = px.line(filtered_df, x="date", y=selected_metric, markers=True, title=title)
    elif chart_type == "Bar Chart":
        fig = px.bar(filtered_df, x="date", y=selected_metric, title=title, color_discrete_sequence=['skyblue'])
    elif chart_type == "Area Chart":
        fig = px.area(filtered_df, x="date", y=selected_metric, title=title, color_discrete_sequence=['lightgreen'])
    elif chart_type == "Scatter Plot":
        trend = "ols" if show_trendline else None
        fig = px.scatter(
            filtered_df,
            x="date",
            y=selected_metric,
            title=title,
            trendline=trend,
            size_max=15,
            color_discrete_sequence=['orange'],
            hover_data=filtered_df.columns
        )

# === 6. Style the chart ===
fig.update_layout(
    xaxis_title="Quarter",
    yaxis_title=selected_metric_label if not multi_metric_labels else "Value",
    title_font_size=18,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=14),
    margin=dict(l=40, r=40, t=60, b=40),
)

st.plotly_chart(fig, use_container_width=True)
