import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta import others
import time

# Page configuration
st.set_page_config(
    page_title="ETF Monthly Return Analysis",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 ETF Monthly Return Analysis Dashboard")
st.markdown("Analyze monthly return patterns, probabilities, and performance metrics across ETFs")

# Sidebar configuration
st.sidebar.header("Configuration")

# ETF selections
BROAD_CORES = ["VOO", "VTI", "VT", "VXUS", "VEA"]
SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
COMMODITIES = ["GLD", "SLV"]

period = st.sidebar.selectbox("Time Period", ["5y", "10y", "max"], index=1)
interval = "1mo"

# Asset class selection
asset_classes = st.sidebar.multiselect(
    "Select Asset Classes",
    ["Broad Core ETFs", "Sector ETFs", "Commodity ETFs"],
    default=["Broad Core ETFs", "Sector ETFs", "Commodity ETFs"]
)

# Cache data loading
@st.cache_data(ttl=3600)
def load_etf_data(tickers, period, interval):
    """Load ETF data from Yahoo Finance"""
    data_dict = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, ticker in enumerate(tickers):
        status_text.text(f"Loading {ticker}... ({idx+1}/{len(tickers)})")
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            if not df.empty:
                df["pct_change"] = df["Close"].pct_change() * 100
                df["cumulative_return"] = others.cumulative_return(df["Close"])
                data_dict[ticker] = df
            else:
                st.warning(f"No data available for {ticker}")
        except Exception as e:
            st.error(f"Error loading {ticker}: {str(e)}")

        progress_bar.progress((idx + 1) / len(tickers))

    status_text.empty()
    progress_bar.empty()
    return data_dict

# Helper functions
def calculate_annual_sharpe(pct_return: pd.Series, risk_free_rate: float = 0.05):
    """Calculate annualized Sharpe ratio"""
    if pct_return.std() == 0:
        return np.nan
    return ((pct_return.mean() - risk_free_rate) / pct_return.std()) * np.sqrt(252)

def calc_positive_prob(column):
    """Calculate probability of positive returns"""
    valid_values = column.dropna()
    if len(valid_values) == 0:
        return np.nan
    positive_count = (valid_values > 0).sum()
    total_count = len(valid_values)
    return (positive_count / total_count) * 100

def calculate_monthly_stats_for_etfs(etf_dict, etf_list):
    """Calculate monthly statistics for all ETFs"""
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']

    prob_results = {}
    avg_return_results = {}

    for etf in etf_list:
        if etf not in etf_dict or etf_dict[etf].empty:
            continue

        df_temp = etf_dict[etf][["pct_change"]].copy()
        df_temp['Year'] = df_temp.index.year
        df_temp['Month'] = df_temp.index.month_name()

        pivot = df_temp.pivot_table(
            values='pct_change',
            index='Year',
            columns='Month',
            aggfunc='first'
        )
        pivot = pivot.reindex(columns=month_order)

        prob_results[etf] = pivot.apply(calc_positive_prob, axis=0)
        avg_return_results[etf] = pivot.mean(axis=0)

    return pd.DataFrame(prob_results).T, pd.DataFrame(avg_return_results).T

def plot_cumulative_return(data_dict, title="Cumulative Return"):
    """Create plotly cumulative return chart"""
    fig = go.Figure()

    for ticker in data_dict:
        if not data_dict[ticker].empty:
            fig.add_trace(
                go.Scatter(
                    x=data_dict[ticker].index, 
                    y=data_dict[ticker]["cumulative_return"], 
                    name=ticker,
                    mode='lines'
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        showlegend=True,
        height=500,
        hovermode='x unified'
    )

    return fig

# Main app logic
# Determine which ETFs to load
selected_tickers = []
if "Broad Core ETFs" in asset_classes:
    selected_tickers.extend(BROAD_CORES)
if "Sector ETFs" in asset_classes:
    selected_tickers.extend(SECTORS)
if "Commodity ETFs" in asset_classes:
    selected_tickers.extend(COMMODITIES)

if not selected_tickers:
    st.warning("Please select at least one asset class from the sidebar.")
    st.stop()

# Load data
with st.spinner("Loading ETF data..."):
    all_data = load_etf_data(selected_tickers, period, interval)

if not all_data:
    st.error("No data loaded. Please check your selections.")
    st.stop()

# Separate by asset class
broad_cores_dict = {k: v for k, v in all_data.items() if k in BROAD_CORES}
sectors_dict = {k: v for k, v in all_data.items() if k in SECTORS}
commodities_dict = {k: v for k, v in all_data.items() if k in COMMODITIES}

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "📈 Cumulative Returns", 
    "🎯 Monthly Probability", 
    "💰 Average Returns",
    "📉 Sharpe Ratios"
])

with tab1:
    st.header("Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total ETFs Analyzed", len(all_data))
    with col2:
        st.metric("Time Period", period.upper())
    with col3:
        st.metric("Data Interval", interval.upper())

    st.subheader("ETFs Included in Analysis")

    if broad_cores_dict:
        st.write("**Broad Core ETFs:**", ", ".join(broad_cores_dict.keys()))
    if sectors_dict:
        st.write("**Sector ETFs:**", ", ".join(sectors_dict.keys()))
    if commodities_dict:
        st.write("**Commodity ETFs:**", ", ".join(commodities_dict.keys()))

with tab2:
    st.header("Cumulative Returns")

    if broad_cores_dict:
        st.subheader("Broad Core ETFs")
        fig = plot_cumulative_return(broad_cores_dict, "Cumulative Return - Broad Core ETFs")
        st.plotly_chart(fig, use_container_width=True)

    if sectors_dict:
        st.subheader("Sector ETFs")
        fig = plot_cumulative_return(sectors_dict, "Cumulative Return - Sector ETFs")
        st.plotly_chart(fig, use_container_width=True)

    if commodities_dict:
        st.subheader("Commodity ETFs")
        fig = plot_cumulative_return(commodities_dict, "Cumulative Return - Commodity ETFs")
        st.plotly_chart(fig, use_container_width=True)

    # Combined view
    if len(all_data) > 1:
        st.subheader("All ETFs Combined")
        fig = plot_cumulative_return(all_data, "Cumulative Return - All ETFs")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Monthly Probability of Positive Returns")

    # Calculate probabilities
    if broad_cores_dict:
        st.subheader("Broad Core ETFs")
        broad_prob, _ = calculate_monthly_stats_for_etfs(broad_cores_dict, list(broad_cores_dict.keys()))
        st.dataframe(broad_prob.round(2).style.background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)

        # Download button
        csv = broad_prob.to_csv().encode('utf-8')
        st.download_button(
            "Download Broad Core Probability Data",
            csv,
            "broad_cores_probability.csv",
            "text/csv"
        )

    if sectors_dict:
        st.subheader("Sector ETFs")
        sectors_prob, _ = calculate_monthly_stats_for_etfs(sectors_dict, list(sectors_dict.keys()))
        st.dataframe(sectors_prob.round(2).style.background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)

        csv = sectors_prob.to_csv().encode('utf-8')
        st.download_button(
            "Download Sector Probability Data",
            csv,
            "sectors_probability.csv",
            "text/csv"
        )

    if commodities_dict:
        st.subheader("Commodity ETFs")
        commodities_prob, _ = calculate_monthly_stats_for_etfs(commodities_dict, list(commodities_dict.keys()))
        st.dataframe(commodities_prob.round(2).style.background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)

        csv = commodities_prob.to_csv().encode('utf-8')
        st.download_button(
            "Download Commodity Probability Data",
            csv,
            "commodities_probability.csv",
            "text/csv"
        )

    # Summary by month
    if len(all_data) > 1:
        st.subheader("Average Probability by Month (All ETFs)")
        all_prob, _ = calculate_monthly_stats_for_etfs(all_data, list(all_data.keys()))
        month_avg_prob = all_prob.mean(axis=0).to_frame(name="Average Probability (%)")

        fig = go.Figure(data=[
            go.Bar(x=month_avg_prob.index, y=month_avg_prob["Average Probability (%)"], 
                   marker_color='lightblue')
        ])
        fig.update_layout(
            title="Average Probability of Positive Return by Month",
            xaxis_title="Month",
            yaxis_title="Probability (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Average Monthly Returns")

    # Calculate average returns
    if broad_cores_dict:
        st.subheader("Broad Core ETFs")
        _, broad_avg = calculate_monthly_stats_for_etfs(broad_cores_dict, list(broad_cores_dict.keys()))
        st.dataframe(broad_avg.round(2).style.background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)

        csv = broad_avg.to_csv().encode('utf-8')
        st.download_button(
            "Download Broad Core Average Returns",
            csv,
            "broad_cores_avg_returns.csv",
            "text/csv"
        )

    if sectors_dict:
        st.subheader("Sector ETFs")
        _, sectors_avg = calculate_monthly_stats_for_etfs(sectors_dict, list(sectors_dict.keys()))
        st.dataframe(sectors_avg.round(2).style.background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)

        csv = sectors_avg.to_csv().encode('utf-8')
        st.download_button(
            "Download Sector Average Returns",
            csv,
            "sectors_avg_returns.csv",
            "text/csv"
        )

    if commodities_dict:
        st.subheader("Commodity ETFs")
        _, commodities_avg = calculate_monthly_stats_for_etfs(commodities_dict, list(commodities_dict.keys()))
        st.dataframe(commodities_avg.round(2).style.background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)

        csv = commodities_avg.to_csv().encode('utf-8')
        st.download_button(
            "Download Commodity Average Returns",
            csv,
            "commodities_avg_returns.csv",
            "text/csv"
        )

    # Summary by month
    if len(all_data) > 1:
        st.subheader("Average Return by Month (All ETFs)")
        _, all_avg = calculate_monthly_stats_for_etfs(all_data, list(all_data.keys()))
        month_avg_return = all_avg.mean(axis=0).to_frame(name="Average Return (%)")

        fig = go.Figure(data=[
            go.Bar(x=month_avg_return.index, y=month_avg_return["Average Return (%)"],
                   marker_color='lightgreen')
        ])
        fig.update_layout(
            title="Average Return by Month",
            xaxis_title="Month",
            yaxis_title="Return (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Sharpe Ratios")

    sharpe_results = {}
    for ticker, data in all_data.items():
        if not data.empty and 'pct_change' in data.columns:
            sharpe = calculate_annual_sharpe(data['pct_change'].dropna())
            sharpe_results[ticker] = sharpe

    if sharpe_results:
        sharpe_df = pd.DataFrame.from_dict(sharpe_results, orient='index', columns=['Sharpe Ratio'])
        sharpe_df = sharpe_df.sort_values('Sharpe Ratio', ascending=False)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure(data=[
                go.Bar(x=sharpe_df.index, y=sharpe_df['Sharpe Ratio'],
                       marker_color='steelblue')
            ])
            fig.update_layout(
                title="Annualized Sharpe Ratios",
                xaxis_title="ETF",
                yaxis_title="Sharpe Ratio",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Rankings")
            st.dataframe(sharpe_df.round(3), use_container_width=True)

            csv = sharpe_df.to_csv().encode('utf-8')
            st.download_button(
                "Download Sharpe Ratios",
                csv,
                "sharpe_ratios.csv",
                "text/csv"
            )

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Data source: Yahoo Finance via yfinance")
st.sidebar.caption("Note: Past performance does not guarantee future results.")
