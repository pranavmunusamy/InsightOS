import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# ─── Setup ─────────────────────────────────────────────────────────────
st.set_page_config("📊 Data Explorer", layout="wide")

st.markdown("""
<style>
.glow {
    background-color: #1e1e1e;
    border-radius: 15px;
    box-shadow: 0 0 15px #14FFEC;
    padding: 20px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="glow"><h1 style="color:#14FFEC;">📊 Data Explorer App</h1></div>', unsafe_allow_html=True)

def generate_3line_insight(col_data, col_name):
    col_data = col_data.dropna()
    mean = col_data.mean()
    median = col_data.median()
    std = col_data.std()
    skew = col_data.skew()
    min_val, max_val = col_data.min(), col_data.max()
    outlier_threshold = mean + 2 * std

    if skew > 1:
        shape = "highly right-skewed"
    elif skew > 0.5:
        shape = "moderately right-skewed"
    elif skew < -1:
        shape = "highly left-skewed"
    elif skew < -0.5:
        shape = "moderately left-skewed"
    else:
        shape = "approximately symmetric"

    return (
        f"- Mean: {mean:.2f}, Median: {median:.2f}, Std Dev: {std:.2f}\n"
        f"- Distribution is {shape}.\n"
        f"- Values range from {min_val:.2f} to {max_val:.2f}, with potential outliers above {outlier_threshold:.2f}."
    )

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if not uploaded:
    st.stop()

ext = Path(uploaded.name).suffix.lower()
df = pd.read_csv(uploaded) if ext == ".csv" else pd.read_excel(uploaded, engine="openpyxl")
if df.empty:
    st.warning("Uploaded file is empty.")
    st.stop()

st.subheader("🔍 Data Preview")
st.dataframe(df.head(), use_container_width=True)
st.markdown(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
st.write("Null Values:")
st.write(df.isnull().sum())

nums = df.select_dtypes(include=np.number).columns.tolist()
cats, dts = [], []
for col in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        dts.append(col)
    elif df[col].dtype == object:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.7:
            df[col] = parsed
            dts.append(col)
        else:
            cats.append(col)
    else:
        if col not in nums + dts:
            cats.append(col)

st.sidebar.title("🎛️ Chart Settings")
show_charts = {
    "Histogram": st.sidebar.checkbox("Histogram"),
    "Line Chart": st.sidebar.checkbox("Line Chart"),
    "Box Plot": st.sidebar.checkbox("Box Plot"),
    "Bar Chart": st.sidebar.checkbox("Bar Chart"),
    "Scatter Plot": st.sidebar.checkbox("Scatter Plot"),
    "Violin Plot": st.sidebar.checkbox("Violin Plot"),
    "Area Chart": st.sidebar.checkbox("Area Chart"),
    "Radar Chart": st.sidebar.checkbox("Radar Chart")
}

chart_tabs = [key for key, val in show_charts.items() if val]
if not chart_tabs:
    st.info("☝️ Enable at least one chart type from the sidebar.")
    st.stop()

tabs = st.tabs(chart_tabs)
pdf_figures = []

for i, chart in enumerate(chart_tabs):
    with tabs[i]:
        if chart == "Histogram":
            sel = st.multiselect("Numeric Columns", nums)
            bins = st.slider("Bins", 5, 100, 20)
            for col in sel:
                fig = px.histogram(df, x=col, nbins=bins)
                st.plotly_chart(fig, use_container_width=True)
                insight = generate_3line_insight(df[col], col)
                st.markdown(insight)
                pdf_figures.append((fig, insight))

        elif chart == "Line Chart":
            if dts:
                dt = st.selectbox("Date Column", dts)
                ycols = st.multiselect("Y-axis", nums)
                for col in ycols:
                    d = df[[dt, col]].dropna().sort_values(dt)
                    fig = px.line(d, x=dt, y=col)
                    st.plotly_chart(fig, use_container_width=True)
                    insight = generate_3line_insight(d[col], col)
                    st.markdown(insight)
                    pdf_figures.append((fig, insight))

        elif chart == "Box Plot":
            cat = st.selectbox("X (Category)", cats, key="box_x")
            num = st.selectbox("Y (Numeric)", nums, key="box_y")
            fig = px.box(df, x=cat, y=num, color=cat)
            st.plotly_chart(fig, use_container_width=True)
            insight = generate_3line_insight(df[num], num)
            st.markdown(insight)
            pdf_figures.append((fig, insight))

        elif chart == "Bar Chart":
            col = st.selectbox("Category Column", cats, key="bar")
            vc = df[col].value_counts().nlargest(10).reset_index()
            vc.columns = ["Category", "Count"]
            fig = px.bar(vc, x="Category", y="Count", color="Category")
            st.plotly_chart(fig, use_container_width=True)
            insight = f"- Most frequent value: {vc.iloc[0]['Category']} with {vc.iloc[0]['Count']} occurrences.\n- Displaying top 10 categories.\n- Useful for categorical dominance."
            st.markdown(insight)
            pdf_figures.append((fig, insight))

        elif chart == "Scatter Plot":
            x = st.selectbox("X", nums, key="scat_x")
            y = st.selectbox("Y", [n for n in nums if n != x], key="scat_y")
            color = st.selectbox("Color By", [None] + cats)
            fig = px.scatter(df, x=x, y=y, color=color)
            st.plotly_chart(fig, use_container_width=True)
            corr = df[[x, y]].corr().iloc[0, 1]
            insight = f"- Scatter between {x} and {y}.\n- Correlation: {corr:.2f}.\n- Potential trend or cluster detection."
            st.markdown(insight)
            pdf_figures.append((fig, insight))

        elif chart == "Violin Plot":
            cat = st.selectbox("Category", cats, key="vcat")
            num = st.selectbox("Numeric", nums, key="vnum")
            fig = px.violin(df, x=cat, y=num, color=cat, box=True, points="all")
            st.plotly_chart(fig, use_container_width=True)
            insight = generate_3line_insight(df[num], num)
            st.markdown(insight)
            pdf_figures.append((fig, insight))

        elif chart == "Area Chart":
            if dts:
                dt = st.selectbox("Date", dts, key="adt")
                ycols = st.multiselect("Y Columns", nums, key="ay")
                for col in ycols:
                    d = df[[dt, col]].dropna().sort_values(dt)
                    fig = px.area(d, x=dt, y=col)
                    st.plotly_chart(fig, use_container_width=True)
                    insight = generate_3line_insight(d[col], col)
                    st.markdown(insight)
                    pdf_figures.append((fig, insight))

        elif chart == "Radar Chart":
            radar_cols = st.multiselect("Select 3–5 Numeric Columns", nums, key="radar")
            if 3 <= len(radar_cols) <= 5:
                means = df[radar_cols].mean().reset_index()
                means.columns = ["Metric", "Value"]
                means = pd.concat([means, means.iloc[[0]]], ignore_index=True)
                fig = go.Figure(data=go.Scatterpolar(
                    r=means["Value"],
                    theta=means["Metric"],
                    fill='toself',
                    name='Average'
                ))
                fig.update_layout(showlegend=True,
                                  polar=dict(radialaxis=dict(visible=True)))
                st.plotly_chart(fig, use_container_width=True)
                insight = f"- Radar chart of selected metrics.\n- Highest mean: {means.iloc[:-1]['Value'].max():.2f}.\n- Visualizes comparative strength across KPIs."
                st.markdown(insight)
                pdf_figures.append((fig, insight))

st.subheader("📤 Export Report")
if pdf_figures:
    if st.button("Generate PDF Report"):
        pdf_buffer = BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            for fig, text in pdf_figures:
                fig_bytes = fig.to_image(format="png")
                img = plt.imread(BytesIO(fig_bytes))
                plt.figure(figsize=(10, 6))
                plt.imshow(img)
                plt.axis("off")
                plt.title(text, fontsize=8)
                pdf.savefig()
                plt.close()
        st.download_button("📥 Download PDF", data=pdf_buffer.getvalue(),
                           file_name="report.pdf", mime="application/pdf")
else:
    st.info("Generate at least one chart to enable PDF export.")
