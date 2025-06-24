import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("MalaysiaUnemploymentRate.csv")
df['Year'] = df['Year'].astype(int)
df = df.sort_values('Year')

# App title
st.title("Malaysia Annual Unemployment Rate (1991 - 2024)")

# === 1. Show full table first ===
st.subheader("Full Data Table")
st.dataframe(df, use_container_width=True)

# === 2. Setup parameters for filtering ===
min_year = df['Year'].min()
max_year = df['Year'].max()

# === 3. Chart Controls ===
st.subheader("Chart Controls")

# Chart type selector
chart_type = st.selectbox(
    "Choose chart type:",
    ["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot"]
)

# Duration selector including "All"
duration_option = st.radio("How many years to display?", ["All", 10, 20, 30], horizontal=True)

# === 4. Filter data based on selection ===
if duration_option == "All":
    start_year = min_year
    end_year = max_year
    filtered_df = df.copy()
else:
    duration = int(duration_option)
    end_year = max_year
    start_year = end_year - duration + 1
    filtered_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# === 5. Show filtered data table ===
st.subheader(f"Filtered Data Table: {start_year} – {end_year}")
st.dataframe(filtered_df, use_container_width=True)

# === 6. Show chart (modern version using Plotly) ===
st.markdown("---")
st.subheader("Unemployment Rate Trend")

title = f"Unemployment Rate: {start_year} – {end_year}"

if chart_type == "Line Chart":
    fig = px.line(filtered_df, x="Year", y="Unemployment Rate", markers=True, title=title)
elif chart_type == "Bar Chart":
    fig = px.bar(filtered_df, x="Year", y="Unemployment Rate", title=title, color_discrete_sequence=['skyblue'])
elif chart_type == "Area Chart":
    fig = px.area(filtered_df, x="Year", y="Unemployment Rate", title=title, color_discrete_sequence=['lightgreen'])
elif chart_type == "Scatter Plot":
    fig = px.scatter(filtered_df, x="Year", y="Unemployment Rate", size_max=15, title=title, color_discrete_sequence=['red'])

# Update layout
fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Unemployment Rate (%)",
    title_font_size=18,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=14),
    margin=dict(l=40, r=40, t=60, b=40),
)

st.plotly_chart(fig, use_container_width=True)
