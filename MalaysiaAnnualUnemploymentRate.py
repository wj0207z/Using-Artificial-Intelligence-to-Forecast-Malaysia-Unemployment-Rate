import streamlit as st
import pandas as pd
import plotly.express as px

# === 1. Load dataset ===
df = pd.read_csv("MalaysiaQuarterlyLabourForce.csv")

# Convert 'date' to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract year
df['year'] = df['date'].dt.year

# === 2. Calculate average unemployment rate per year with count of quarters ===
annual_stats = df.groupby('year')['u_rate'].agg(['mean', 'count']).reset_index()
annual_stats.columns = ['year', 'Average Unemployment Rate', 'Number of Quarters']
annual_stats['Average Unemployment Rate'] = annual_stats['Average Unemployment Rate'].round(2)

# === 3. App Title ===
st.title("📉 Malaysia Annual Average Unemployment Rate")

# === 4. Controls ===
st.subheader("Chart Controls")

# Chart type
chart_type = st.selectbox("📊 Choose chart type:", ["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot"])

# Trendline toggle
show_trendline = st.checkbox("📈 Add trendline (OLS) — for Scatter Plot only")

# Year range selector
min_year = int(annual_stats['year'].min())
max_year = int(annual_stats['year'].max())
start_year, end_year = st.slider("📆 Select year range:", min_year, max_year, (min_year, max_year))

# Filter by year range
filtered_df = annual_stats[(annual_stats['year'] >= start_year) & (annual_stats['year'] <= end_year)]

# === 5. Summary Stats ===
if not filtered_df.empty:
    highest = filtered_df.loc[filtered_df['Average Unemployment Rate'].idxmax()]
    lowest = filtered_df.loc[filtered_df['Average Unemployment Rate'].idxmin()]
    st.markdown(f"🔺 **Highest Rate**: {highest['Average Unemployment Rate']}% in {int(highest['year'])}")
    st.markdown(f"🔻 **Lowest Rate**: {lowest['Average Unemployment Rate']}% in {int(lowest['year'])}")
else:
    st.warning("No data available for selected year range.")

# === 6. Data Table ===
st.subheader(f"📄 Average Annual Unemployment Rate ({start_year} – {end_year})")

table_df = filtered_df.reset_index(drop=True)
table_df.index = table_df.index + 1  # Start index from 1
st.dataframe(table_df, use_container_width=True)

# Download CSV
csv = table_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download as CSV", data=csv, file_name="annual_unemployment.csv", mime="text/csv")

# === 7. Chart ===
st.markdown("---")
st.subheader("📈 Unemployment Rate Trend")

title = f"Annual Average Unemployment Rate ({start_year} – {end_year})"

if chart_type == "Line Chart":
    fig = px.line(
        filtered_df, x="year", y="Average Unemployment Rate",
        markers=True, title=title
    )
elif chart_type == "Bar Chart":
    fig = px.bar(
        filtered_df, x="year", y="Average Unemployment Rate",
        title=title, color_discrete_sequence=['skyblue']
    )
elif chart_type == "Area Chart":
    fig = px.area(
        filtered_df, x="year", y="Average Unemployment Rate",
        title=title, color_discrete_sequence=['lightgreen']
    )
elif chart_type == "Scatter Plot":
    trend = "ols" if show_trendline else None
    fig = px.scatter(
        filtered_df, x="year", y="Average Unemployment Rate",
        title=title, trendline=trend, color_discrete_sequence=['orange'], size_max=15
    )

fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Unemployment Rate (%)",
    title_font_size=18,
    font=dict(size=14),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=40, r=40, t=60, b=40),
)

st.plotly_chart(fig, use_container_width=True)
